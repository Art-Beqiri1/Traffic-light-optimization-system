"""Shared live-demo controller: AI, fixed timer, manual and latched fallback.

Signal timing is independent of AI/training threads. No physical signal I/O.
The offline training/benchmark environment keeps its original timing rules.
"""
from __future__ import annotations

import copy
import threading
import time
from collections import deque
from numbers import Integral
from typing import Callable, Dict, Optional, Tuple

from network_env import TrafficNetwork, DiurnalArrivalProfile, Phase_NS, Phase_EW, MAX_GREEN
from controllers import MaxPressurePolicy
from signal_engine import SignalSequencer

NodeId = Tuple[int, int]
DEFAULT_PROFILE = DiurnalArrivalProfile(base_rates={"N": .22, "S": .19, "E": .11, "W": .09})
DEFAULT_CONTROL = {"mode": "ai", "fixed_ns_seconds": 25, "fixed_ew_seconds": 25,
                   "yellow_seconds": 3, "all_red_seconds": 2}


def validate_control(data, current=None):
    if not isinstance(data, dict):
        raise ValueError("Control settings must be an object")
    allowed = set(DEFAULT_CONTROL) | {"manual_phase"}
    if set(data) - allowed:
        raise ValueError("Unknown control setting")
    result = dict(current or DEFAULT_CONTROL)
    result.update(data)
    if result["mode"] not in ("ai", "manual", "fixed"):
        raise ValueError("mode must be ai, manual or fixed")
    for key, low, high in (("fixed_ns_seconds", 7, 60), ("fixed_ew_seconds", 7, 60),
                           ("yellow_seconds", 1, 10), ("all_red_seconds", 1, 10)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
    if "manual_phase" in result:
        if result["manual_phase"] not in ("NS_GREEN", "EW_GREEN", "ALL_RED"):
            raise ValueError("manual_phase must be NS_GREEN, EW_GREEN or ALL_RED")
        if result["mode"] != "manual":
            raise ValueError("Manual light commands require manual mode")
    return result


class LiveTrafficController:
    def __init__(self, rows=2, cols=2, tick_seconds=1.0, camera_node=(0, 0),
                 profile=None, dqn_episodes=120, dqn_train_window=600,
                 history_len=600, get_live_queues=None, get_camera_health=None,
                 control_config=None, clock=time.monotonic, ai_timeout=3.0):
        self.rows, self.cols = rows, cols
        self.tick_seconds, self.camera_node = tick_seconds, camera_node
        self.profile = profile or DEFAULT_PROFILE
        self.dqn_episodes, self.dqn_train_window = dqn_episodes, dqn_train_window
        self.get_live_queues, self.get_camera_health = get_live_queues, get_camera_health
        self.node_ids = [(r, c) for r in range(rows) for c in range(cols)]
        self.net = TrafficNetwork(rows, cols, self.profile, duration=10**9, seed=7)
        self.net.reset()
        self.bootstrap_policy = MaxPressurePolicy()
        self.trained_policy = None
        self.training_status = "Max-Pressure ready; Double DQN trains on startup"
        self.training_progress = 0.0
        self.control = validate_control(control_config or {})
        self.manual_target = None  # manual startup always holds all red
        self.fallback_reason = ""
        self._clock, self.ai_timeout = clock, ai_timeout
        now = clock()
        self._now = now
        self.signals = {rc: SignalSequencer(now, self.control["yellow_seconds"], self.control["all_red_seconds"])
                        for rc in self.node_ids}
        self._lock = threading.RLock()
        self._running = False
        self._stop_event = threading.Event()
        self._sim_thread = self._train_thread = self._ai_thread = None
        self._ai_result = None
        self._ai_started = None
        self._ai_epoch = 0
        self._ai_targets = {}
        self._last_step = now - tick_seconds
        self._tick_count = self._switch_count = 0
        self.history = deque(maxlen=history_len)

    def start(self):
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        if self._train_thread is None:
            self._train_thread = threading.Thread(target=self._train_loop, daemon=True)
            self._train_thread.start()

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._sim_thread and self._sim_thread is not threading.current_thread():
            self._sim_thread.join(timeout=2)

    def _train_loop(self):
        try:
            # Optional for manual/fixed operation: a missing/broken torch must
            # not prevent the app or its controls from starting.
            from agents import train_dqn, DQNPolicy
            def factory(seed=0):
                return TrafficNetwork(self.rows, self.cols, self.profile,
                                      duration=self.dqn_train_window, seed=seed)
            with self._lock:
                self.training_status = "Training Double DQN; AI currently uses Max-Pressure"
            q_net = train_dqn(factory, len(self.node_ids), self.node_ids,
                              duration=self.dqn_train_window, episodes=self.dqn_episodes, verbose=False)
            with self._lock:
                self.trained_policy = DQNPolicy(q_net, self.node_ids)
                self.training_status = "Double DQN ready (used only in AI mode)"
                self.training_progress = 1.0
        except Exception as error:
            with self._lock:
                self.training_status = f"DQN unavailable ({type(error).__name__}); AI uses Max-Pressure"

    def configure(self, data):
        with self._lock:
            new = validate_control(data, self.control)
            old_mode = self.control["mode"]
            phase = new.pop("manual_phase", None)
            self.control = new
            if "mode" in data:
                self.fallback_reason = ""
                self._ai_epoch += 1  # discard in-flight decisions from an earlier mode
                self._ai_targets.clear()
                if new["mode"] == "manual" and old_mode != "manual":
                    sig = self.signals[self.camera_node]
                    self.manual_target = sig.phase if sig.stage == "GREEN" else None
            if phase is not None:
                self.manual_target = {"NS_GREEN": Phase_NS, "EW_GREEN": Phase_EW, "ALL_RED": None}[phase]
            for sig in self.signals.values():
                sig.yellow, sig.all_red = new["yellow_seconds"], new["all_red_seconds"]
            return self.control_snapshot()

    def _fallback(self, reason):
        if not self.fallback_reason:
            self.fallback_reason = reason
            self._ai_epoch += 1
            self._ai_targets.clear()

    def _submit_ai(self, now):
        # At most ONE worker, even if inference hangs. It never owns the live
        # network or control lock while calling third-party policy code.
        if self._ai_thread is not None and self._ai_thread.is_alive():
            return
        net = copy.deepcopy(self.net)
        policy = self.trained_policy or self.bootstrap_policy
        epoch = self._ai_epoch
        self._ai_started = now
        self._ai_result = None
        def work():
            try:
                actions = policy.act(net)
                if not isinstance(actions, dict) or set(actions) != set(self.node_ids):
                    raise ValueError("incomplete AI actions")
                if any(not isinstance(a, Integral) or a not in (0, 1) for a in actions.values()):
                    raise ValueError("invalid AI actions")
                # Store absolute phase requests so a repeated result cannot
                # reverse a just-completed phase change.
                targets = {rc: (1 - net.nodes[rc].phase if actions[rc] else net.nodes[rc].phase)
                           for rc in self.node_ids}
                error = None
            except Exception as exc:
                targets, error = {}, f"AI decision failed ({type(exc).__name__})"
            with self._lock:
                self._ai_result = (epoch, targets, error)
        self._ai_thread = threading.Thread(target=work, daemon=True)
        self._ai_thread.start()

    def _sim_loop(self):
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                with self._lock:
                    self._fallback(f"Control update error ({type(exc).__name__})")
            self._stop_event.wait(.1)

    def _tick(self, now=None):
        with self._lock:
            now = self._clock() if now is None else now
            self._now = now
            live = None
            try:
                if self.get_live_queues:
                    live = self.get_live_queues()
                    if set(live) != {"N", "S", "E", "W"} or any(
                            not isinstance(v, Integral) or v < 0 for v in live.values()):
                        raise ValueError("invalid live queues")
                    self.net.nodes[self.camera_node].queues.update(live)
                if self.get_camera_health:
                    health = self.get_camera_health()
                    if self.control["mode"] == "ai" and health["configured"] and not health["healthy"]:
                        self._fallback(health["reason"])
            except Exception as exc:
                live = None
                if self.control["mode"] == "ai":
                    self._fallback(f"Camera data unavailable ({type(exc).__name__})")

            if self.control["mode"] == "ai" and not self.fallback_reason:
                if self._ai_result is not None:
                    epoch, targets, error = self._ai_result
                    self._ai_result = None
                    if epoch == self._ai_epoch:
                        if error:
                            self._fallback(error)
                        else:
                            self._ai_targets = targets
                if self._ai_thread and self._ai_thread.is_alive() and now - self._ai_started >= self.ai_timeout:
                    self._fallback("AI decision timed out; fixed timer active")
                if not self.fallback_reason and now - self._last_step >= self.tick_seconds:
                    self._submit_ai(now)

            effective = self.effective_mode
            for rc, sig in self.signals.items():
                age = now - sig.started
                if effective == "manual":
                    desired = self.manual_target
                elif effective == "fixed":
                    duration = self.control["fixed_ns_seconds" if sig.phase == Phase_NS else "fixed_ew_seconds"]
                    desired = ((1 - sig.phase) if age >= duration else sig.phase) if sig.stage == "GREEN" else sig.desired
                    if desired is None:
                        desired = sig.phase
                else:
                    desired = self._ai_targets.get(rc, sig.desired)
                    if desired is None:
                        desired = sig.phase
                    if sig.stage == "GREEN" and age >= MAX_GREEN:
                        desired = 1 - sig.phase
                # Once yellow starts, an AI fluctuation may not cancel the
                # transition. Operator commands may select the next destination.
                if effective != "manual" and sig.stage != "GREEN":
                    desired = sig.desired if sig.desired is not None else sig.phase
                old_stage = sig.stage
                sig.advance(now, desired)
                if old_stage == "GREEN" and sig.stage == "YELLOW":
                    self._switch_count += 1
                node = self.net.nodes[rc]
                node.phase = sig.phase
                node.clearing = 0 if sig.stage == "GREEN" else 1
                node.time_in_phase = max(0, now - sig.started) if sig.stage == "GREEN" else 0

            if now - self._last_step >= self.tick_seconds:
                # No catch-up loop: long pauses never skip visible stages.
                self._last_step = now
                self.net.step({}, external_signals=True,
                              live_queues={self.camera_node: live} if live is not None else None)
                self._tick_count += 1
                avg_wait = self.net.cumulative_wait_total() / max(1, self.net.arrived_total())
                self.history.append({"t": self._tick_count, "queue": self.net.total_queue_all(),
                                     "avg_wait": round(avg_wait, 2)})
                if live is not None:
                    self.net.nodes[self.camera_node].queues.update(live)

    @property
    def effective_mode(self):
        return "fixed" if self.control["mode"] == "ai" and self.fallback_reason else self.control["mode"]

    def control_snapshot(self):
        with self._lock:
            return {**self.control, "effective_mode": self.effective_mode,
                    "fallback_active": bool(self.fallback_reason), "fallback_reason": self.fallback_reason,
                    "manual_phase": "ALL_RED" if self.manual_target is None else
                        ("NS_GREEN" if self.manual_target == Phase_NS else "EW_GREEN"),
                    "scope": "all displayed intersections; cameras feed node 0-0"}

    def get_camera_signal(self):
        with self._lock:
            node = self.net.nodes[self.camera_node]
            return {**self.signals[self.camera_node].snapshot(self._now),
                    "queue_ns": node.ns_queue(), "queue_ew": node.ew_queue()}

    def snapshot(self):
        with self._lock:
            mode = self.effective_mode
            name = {"manual": "Manual", "fixed": "Fixed timer"}.get(mode)
            if name is None:
                name = (self.trained_policy or self.bootstrap_policy).name
            if self.fallback_reason:
                name += " (fallback)"
            intersections = []
            for rc in self.node_ids:
                n = self.net.nodes[rc]
                intersections.append({"id": f"{rc[0]}-{rc[1]}", "row": rc[0], "col": rc[1],
                    "is_camera_node": rc == self.camera_node,
                    **self.signals[rc].snapshot(self._now), "queue_ns": n.ns_queue(),
                    "queue_ew": n.ew_queue(), "queues": dict(n.queues)})
            recent = list(self.history)[-120:]
            return {"controller": name, "training_status": self.training_status,
                "training_progress": self.training_progress, "control": self.control_snapshot(),
                "grid": {"rows": self.rows, "cols": self.cols},
                "camera_node": f"{self.camera_node[0]}-{self.camera_node[1]}",
                "camera_signal": self.get_camera_signal(), "intersections": intersections,
                "efficiency": {
                    "avg_wait_per_vehicle_s": round(self.net.cumulative_wait_total() / max(1, self.net.arrived_total()), 2),
                    "avg_queue_recent": round(sum(h["queue"] for h in recent) / len(recent), 2) if recent else 0,
                    "total_queue_now": self.net.total_queue_all(), "vehicles_served": self.net.served_total(),
                    "vehicles_arrived": self.net.arrived_total(), "phase_switches": self._switch_count,
                    "ticks": self._tick_count}, "history": recent}
