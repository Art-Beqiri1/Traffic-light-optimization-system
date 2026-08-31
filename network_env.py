"""
FlowMind — Multi-Intersection Network Environment
=====================================================

Extends the original single-intersection simulator (simulation.py) to a
small RxC grid of coupled intersections, so that coordination between
signals actually matters (a bad decision at one intersection floods the
next one downstream — a real "network effect", not just N independent
copies of the same problem).

Grid layout (rows x cols), each cell is one 4-way intersection with the
same phase structure as before (NS green / EW green, MIN_GREEN,
MAX_GREEN, ALL_RED_CLEARANCE safety floors).

Approach naming = direction traffic is ARRIVING FROM (standard traffic-
engineering convention):
  - "W" queue = vehicles arriving from the west, headed east
  - "E" queue = vehicles arriving from the east, headed west
  - "N" queue = vehicles arriving from the north, headed south
  - "S" queue = vehicles arriving from the south, headed north

Coupling: when a node discharges vehicles on an approach, those vehicles
travel one hop and become next-tick ARRIVALS on the corresponding
approach of the neighboring node in that direction of travel:
  - discharge of node(r,c)'s W-approach -> arrival at node(r,c+1)'s W-approach
    (or exits the grid if c is already the last column)
  - discharge of node(r,c)'s E-approach -> arrival at node(r,c-1)'s E-approach
    (or exits the grid if c == 0)
  - discharge of node(r,c)'s N-approach -> arrival at node(r+1,c)'s N-approach
    (or exits south if r is the last row)
  - discharge of node(r,c)'s S-approach -> arrival at node(r-1,c)'s S-approach
    (or exits north if r == 0)

External Poisson arrivals (the rush-hour ArrivalProfile) only enter at the
grid's true boundary approaches (the ones that don't have an upstream
neighbor to feed them).

This is still a lightweight, fast-to-train stand-in for the SUMO digital
twin described in the design document — same idea, scaled from 1 to
RxC intersections so DQN/PPO/multi-agent RL have something a tabular
per-intersection Q-table can't easily handle (a joint state space that
grows exponentially with the number of intersections).
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

DISCHARGE_RATE = 1
MIN_GREEN = 7
ALL_RED_CLEARANCE = 2
MAX_GREEN = 60

Phase_NS, Phase_EW = 0, 1


@dataclass
class ArrivalProfile:
    """Original single-episode profile: one rush-hour bulge centered partway
    through whatever `duration` the episode happens to be. Fine for short,
    fixed-length training/eval episodes; not meaningful for multi-day runs
    (see DiurnalArrivalProfile below for that)."""
    base_rates: Dict[str, float]

    def rate_at(self, t: int, duration: int) -> Dict[str, float]:
        peak_pos = 0.6 * duration
        spread = duration / 3.2
        bulge = math.exp(-((t - peak_pos) ** 2) / (2 * spread ** 2))
        multiplier = 0.55 + 1.3 * bulge
        return {k: v * multiplier for k, v in self.base_rates.items()}


@dataclass
class DiurnalArrivalProfile:
    """A realistic day/night, weekday/weekend traffic-demand curve, driven
    by ABSOLUTE simulated time (seconds since t=0 of the whole run), not
    time-since-episode-start. This is what makes multi-day evaluation
    meaningful: hour 8 is always the morning rush, hour 2 is always quiet,
    no matter how long the run is or where an episode's window happens to
    start.

    Shape per day: a low overnight floor, a morning peak (~07:30-09:00),
    a moderate midday plateau, and a broader, slightly heavier evening
    peak (~16:30-18:30) -- matching the commuter-into-city-then-home
    pattern described for Prishtina in the design document. Every 6th and
    7th day (a "weekend") gets a flattened, lower-amplitude curve.
    """
    base_rates: Dict[str, float]
    weekend_scale: float = 0.6
    night_floor: float = 0.12

    def _hour_multiplier(self, hour: float, is_weekend: bool) -> float:
        def gauss(h, center, width, height):
            return height * math.exp(-((h - center) ** 2) / (2 * width ** 2))

        morning = gauss(hour, 8.0, 1.1, 1.35)
        midday = gauss(hour, 13.0, 2.5, 0.55)
        evening = gauss(hour, 17.5, 1.6, 1.55)
        m = self.night_floor + morning + midday + evening
        if is_weekend:
            # weekends: flatter, lower peaks, demand shifts a bit later
            m = self.night_floor + 0.5 * gauss(hour, 12.5, 3.2, 0.9)
        return m

    def rate_at(self, t_abs: int, duration: int = None) -> Dict[str, float]:
        day = (t_abs // 86400) % 7
        hour = (t_abs % 86400) / 3600.0
        is_weekend = day >= 5
        mult = self._hour_multiplier(hour, is_weekend)
        if is_weekend:
            mult *= self.weekend_scale
        return {k: v * mult for k, v in self.base_rates.items()}


class _Node:
    """One intersection's queues + signal timing state."""

    __slots__ = ("queues", "phase", "time_in_phase", "clearing",
                 "total_arrived", "total_served", "cumulative_wait",
                 "vehicles_cleared", "last_discharge")

    def __init__(self):
        self.queues = {"N": 0, "S": 0, "E": 0, "W": 0}
        self.phase = Phase_NS
        self.time_in_phase = 0
        self.clearing = 0
        self.total_arrived = {"N": 0, "S": 0, "E": 0, "W": 0}
        self.total_served = {"N": 0, "S": 0, "E": 0, "W": 0}
        self.cumulative_wait = 0
        self.vehicles_cleared = 0
        self.last_discharge = {"N": 0, "S": 0, "E": 0, "W": 0}

    def ns_queue(self):
        return self.queues["N"] + self.queues["S"]

    def ew_queue(self):
        return self.queues["E"] + self.queues["W"]

    def total_queue(self):
        return sum(self.queues.values())

    def green_approaches(self):
        if self.clearing > 0:
            return ()
        return ("N", "S") if self.phase == Phase_NS else ("E", "W")

    def can_switch(self):
        return self.time_in_phase >= MIN_GREEN

    def must_switch(self):
        return self.time_in_phase >= MAX_GREEN


def _bin(q: int) -> int:
    edges = [0, 1, 4, 8, 14, 21]
    for i in range(len(edges) - 1, -1, -1):
        if q >= edges[i]:
            return i
    return 0


class TrafficNetwork:
    """A coupled RxC grid of intersections. Vectorized-ish, pure Python
    (fast enough for RL training at this scale: a handful of nodes,
    thousands of ticks per episode)."""

    def __init__(self, rows: int, cols: int, profile: ArrivalProfile,
                 duration: int, seed: int, start_time: int = 0):
        self.rows, self.cols = rows, cols
        self.start_time = start_time
        self.n_nodes = rows * cols
        self.node_ids: List[Tuple[int, int]] = [(r, c) for r in range(rows) for c in range(cols)]
        self.idx = {rc: i for i, rc in enumerate(self.node_ids)}
        self.nodes: Dict[Tuple[int, int], _Node] = {rc: _Node() for rc in self.node_ids}
        self.profile = profile
        self.duration = duration
        self.rng = random.Random(seed)
        self.t = 0
        # per-tick "pending arrivals" produced by this tick's discharge,
        # applied next tick (keeps the update causal/simple)
        self._pending: Dict[Tuple[int, int], Dict[str, int]] = {
            rc: {"N": 0, "S": 0, "E": 0, "W": 0} for rc in self.node_ids
        }

    # ---- boundary helpers ----
    def _is_external(self, r, c, approach) -> bool:
        if approach == "W":
            return c == 0
        if approach == "E":
            return c == self.cols - 1
        if approach == "N":
            return r == 0
        if approach == "S":
            return r == self.rows - 1
        return True

    def _poisson(self, lam: float) -> int:
        k = 0
        p = math.exp(-lam)
        cum = p
        u = self.rng.random()
        while u > cum and k < 6:
            k += 1
            p *= lam / k
            cum += p
        return k

    def reset(self):
        for rc in self.node_ids:
            self.nodes[rc] = _Node()
        self._pending = {rc: {"N": 0, "S": 0, "E": 0, "W": 0} for rc in self.node_ids}
        self.t = 0
        return self._global_state()

    # ---- core tick ----
    def step(self, actions: Dict[Tuple[int, int], int], *, external_signals=False, live_queues=None):
        """actions: {(r,c): 0(keep)/1(switch)} raw intent; safety layer
        (min/max green) is enforced here, same as the single-intersection
        version. Returns (obs, node_rewards, done, info)."""
        t = self.t
        rates_ext = self.profile.rate_at(self.start_time + t, self.duration)

        # 1) apply arrivals (external Poisson at boundary, pending from
        #    last tick's discharge everywhere else)
        for rc in self.node_ids:
            r, c = rc
            node = self.nodes[rc]
            for a in ("N", "S", "E", "W"):
                if self._is_external(r, c, a):
                    arr = self._poisson(rates_ext[a])
                else:
                    arr = self._pending[rc][a]
                node.queues[a] += arr
                node.total_arrived[a] += arr

        # Live camera queues replace synthetic demand at the observed node.
        for rc, queues in (live_queues or {}).items():
            self.nodes[rc].queues.update(queues)

        # reset pending for this tick's discharge to be filled below
        new_pending = {rc: {"N": 0, "S": 0, "E": 0, "W": 0} for rc in self.node_ids}

        # 2) apply signal decisions (safety layer) + discharge
        switched_flags = {}
        for rc in self.node_ids:
            node = self.nodes[rc]
            act = actions.get(rc, 0)
            if external_signals:
                # The live wall-clock sequencer owns phase/clearance entirely.
                switched_flags[rc] = False
            elif node.clearing > 0:
                node.clearing -= 1
                switched_flags[rc] = False
            else:
                switch = bool(act)
                if node.time_in_phase < MIN_GREEN:
                    switch = False          # illegal, safety floor
                elif node.time_in_phase >= MAX_GREEN:
                    switch = True           # forced, no starving
                if switch:
                    node.phase = Phase_EW if node.phase == Phase_NS else Phase_NS
                    node.time_in_phase = 0
                    node.clearing = ALL_RED_CLEARANCE
                else:
                    node.time_in_phase += 1
                switched_flags[rc] = switch

            green = node.green_approaches()
            for a in green:
                served = min(node.queues[a], DISCHARGE_RATE)
                node.queues[a] -= served
                node.total_served[a] += served
                node.vehicles_cleared += served
                node.last_discharge[a] = served

            node.cumulative_wait += node.total_queue()

        # 3) route this tick's discharge to downstream neighbors (or sink)
        for rc in self.node_ids:
            r, c = rc
            node = self.nodes[rc]
            w = node.last_discharge["W"]
            if w and c + 1 < self.cols:
                new_pending[(r, c + 1)]["W"] += w
            e = node.last_discharge["E"]
            if e and c - 1 >= 0:
                new_pending[(r, c - 1)]["E"] += e
            n = node.last_discharge["N"]
            if n and r + 1 < self.rows:
                new_pending[(r + 1, c)]["N"] += n
            s = node.last_discharge["S"]
            if s and r - 1 >= 0:
                new_pending[(r - 1, c)]["S"] += s
            node.last_discharge = {"N": 0, "S": 0, "E": 0, "W": 0}

        self._pending = new_pending
        self.t += 1
        done = self.t >= self.duration

        rewards = {}
        for rc in self.node_ids:
            node = self.nodes[rc]
            rewards[rc] = -float(node.total_queue()) - (6.0 if switched_flags[rc] else 0.0)

        info = {"switched": switched_flags}
        return self._global_state(), rewards, done, info

    # ---- observations ----
    def local_features(self, rc) -> list:
        """5 continuous features for one node: used by both centralized
        (concatenated) and decentralized (per-agent) policies."""
        node = self.nodes[rc]
        return [
            min(node.ns_queue(), 40) / 40.0,
            min(node.ew_queue(), 40) / 40.0,
            float(node.phase),
            min(node.time_in_phase, MAX_GREEN) / MAX_GREEN,
            1.0 if node.can_switch() else 0.0,
        ]

    def _global_state(self) -> list:
        s = []
        for rc in self.node_ids:
            s.extend(self.local_features(rc))
        return s

    def legal_action(self, rc, intended: int) -> int:
        node = self.nodes[rc]
        if node.clearing > 0:
            return 0
        if node.time_in_phase < MIN_GREEN:
            return 0
        if node.time_in_phase >= MAX_GREEN:
            return 1
        return intended

    def downstream_queue(self, rc, phase) -> int:
        """Sum of the corresponding queue at the immediate downstream
        neighbor(s) for a given phase — used by Max-Pressure. 0 if the
        approach exits the grid (no back-pressure from outside)."""
        r, c = rc
        total = 0
        if phase == Phase_NS:
            if r + 1 < self.rows:
                total += self.nodes[(r + 1, c)].queues["N"]
            if r - 1 >= 0:
                total += self.nodes[(r - 1, c)].queues["S"]
        else:
            if c + 1 < self.cols:
                total += self.nodes[(r, c + 1)].queues["W"]
            if c - 1 >= 0:
                total += self.nodes[(r, c - 1)].queues["E"]
        return total

    # ---- aggregate metrics ----
    def total_queue_all(self) -> int:
        return sum(n.total_queue() for n in self.nodes.values())

    def served_total(self) -> int:
        return sum(n.vehicles_cleared for n in self.nodes.values())

    def arrived_total(self) -> int:
        return sum(sum(n.total_arrived.values()) for n in self.nodes.values())

    def cumulative_wait_total(self) -> int:
        return sum(n.cumulative_wait for n in self.nodes.values())
