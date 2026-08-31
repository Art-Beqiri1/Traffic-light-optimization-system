"""
Rule-based controllers for the multi-intersection network:

  - FixedTimerPolicy : today's status quo — switch on a fixed clock,
                       blind to demand.
  - MaxPressurePolicy: the classical (non-learning) traffic-engineering
                       algorithm. At every node, computes "pressure" for
                       each phase = queue waiting to be served by that
                       phase minus the queue already building up
                       downstream of it (so it won't dump vehicles into
                       an intersection that's already jammed). Switches
                       to whichever phase currently has higher pressure.
                       This is the standard baseline that DQN/PPO/MARL
                       papers compare against — it needs no training at
                       all, just live queue counts.
"""
from network_env import TrafficNetwork, Phase_NS, Phase_EW


class FixedTimerPolicy:
    name = "Fixed-timer (baseline)"

    def __init__(self, green_duration: int = 25):
        self.green_duration = green_duration

    def act(self, net: TrafficNetwork) -> dict:
        actions = {}
        for rc in net.node_ids:
            node = net.nodes[rc]
            actions[rc] = 1 if node.time_in_phase >= self.green_duration else 0
        return actions


class MaxPressurePolicy:
    name = "Max-Pressure"

    def act(self, net: TrafficNetwork) -> dict:
        actions = {}
        for rc in net.node_ids:
            node = net.nodes[rc]
            p_ns = node.ns_queue() - net.downstream_queue(rc, Phase_NS)
            p_ew = node.ew_queue() - net.downstream_queue(rc, Phase_EW)
            current_pressure = p_ns if node.phase == Phase_NS else p_ew
            other_pressure = p_ew if node.phase == Phase_NS else p_ns
            actions[rc] = 1 if other_pressure > current_pressure else 0
        return actions


def run_episode(net_factory, policy, duration: int):
    """Runs one full episode of `policy` against a freshly-built network,
    returns a dict of summary metrics + time series (for plotting)."""
    net = net_factory()
    queue_series = []
    switch_count = 0
    for _ in range(duration):
        actions = policy.act(net)
        _, _, done, info = net.step(actions)
        switch_count += sum(1 for v in info["switched"].values() if v)
        queue_series.append(net.total_queue_all())
    arrived = net.arrived_total()
    served = net.served_total()
    wait = net.cumulative_wait_total()
    return {
        "controller": policy.name,
        "avg_wait_per_vehicle_s": round(wait / arrived, 2) if arrived else 0.0,
        "max_queue": max(queue_series) if queue_series else 0,
        "avg_queue": round(sum(queue_series) / len(queue_series), 2) if queue_series else 0.0,
        "vehicles_served": served,
        "vehicles_arrived": arrived,
        "phase_switches": switch_count,
        "queue_series": queue_series,
    }
