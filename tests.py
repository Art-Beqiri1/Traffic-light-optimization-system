"""
Sanity tests for the FlowMind multi-intersection RL project.

Deliberately NOT testing "does the trained policy beat the baseline" --
that's a research question with a real, honest answer that varies by
training budget (see README), not a pass/fail unit test. What IS tested
here is the stuff a bug would silently break without necessarily showing
up as an obviously bad-looking chart: the safety layer's hard
constraints, the simulation's flow bookkeeping, the traffic profile's
basic shape, and the DQN action encoding round-trip.

Run with:
    python -m unittest tests.py -v
or just:
    python tests.py
"""
import random
import unittest

from network_env import (
    TrafficNetwork, ArrivalProfile, DiurnalArrivalProfile,
    MIN_GREEN, MAX_GREEN,
)
from controllers import FixedTimerPolicy, MaxPressurePolicy
from agents import decode_joint_action


PROFILE = ArrivalProfile(base_rates={"N": 0.22, "S": 0.19, "E": 0.11, "W": 0.09})


class TestFlowConservation(unittest.TestCase):
    """At every node, cumulative arrivals minus cumulative departures must
    equal the current queue. Summed across all nodes, that means
    arrived_total() - served_total() == total_queue_all() at every tick --
    a bug in the routing/discharge logic (e.g. a lost or duplicated
    vehicle) would break this identity even if the resulting chart still
    looked plausible at a glance."""

    def test_conservation_holds_every_tick_fixed_timer(self):
        net = TrafficNetwork(2, 2, PROFILE, duration=500, seed=1)
        net.reset()
        policy = FixedTimerPolicy(green_duration=25)
        for _ in range(500):
            actions = policy.act(net)
            net.step(actions)
            balance = net.arrived_total() - net.served_total()
            self.assertEqual(balance, net.total_queue_all(),
                              "arrivals - served should equal current queue")

    def test_conservation_holds_every_tick_max_pressure(self):
        net = TrafficNetwork(2, 2, PROFILE, duration=500, seed=2)
        net.reset()
        policy = MaxPressurePolicy()
        for _ in range(500):
            actions = policy.act(net)
            net.step(actions)
            balance = net.arrived_total() - net.served_total()
            self.assertEqual(balance, net.total_queue_all())


class TestSafetyLayer(unittest.TestCase):
    """The safety layer (min/max green) is a hard constraint enforced by
    the environment itself, independent of what any controller -- learned
    or not -- requests. These tests hammer it with a policy that always
    tries to switch (or never does), which is the adversarial case most
    likely to expose an off-by-one in the min/max green logic."""

    def test_forced_switch_by_max_green(self):
        net = TrafficNetwork(1, 1, PROFILE, duration=300, seed=4)
        net.reset()
        node = net.nodes[(0, 0)]
        # never voluntarily switch -- the safety layer must force one anyway
        switched_ticks = []
        for t in range(300):
            prev_phase = node.phase
            net.step({(0, 0): 0})
            if node.phase != prev_phase:
                switched_ticks.append(t)
        self.assertTrue(len(switched_ticks) > 0,
                         "a policy that never requests a switch must still be forced to switch")
        gaps = [switched_ticks[i + 1] - switched_ticks[i] for i in range(len(switched_ticks) - 1)]
        for gap in gaps:
            self.assertLessEqual(gap, MAX_GREEN + 4,
                                  "forced switch should happen at or shortly after MAX_GREEN")

    def test_switch_requests_are_throttled_to_at_least_min_green_apart(self):
        net = TrafficNetwork(1, 1, PROFILE, duration=300, seed=3)
        net.reset()
        node = net.nodes[(0, 0)]
        switched_ticks = []
        for t in range(300):
            prev_phase = node.phase
            net.step({(0, 0): 1})  # always request a switch
            if node.phase != prev_phase:
                switched_ticks.append(t)
        gaps = [switched_ticks[i + 1] - switched_ticks[i] for i in range(len(switched_ticks) - 1)]
        for gap in gaps:
            self.assertGreaterEqual(gap, MIN_GREEN,
                                     "should never switch again before MIN_GREEN elapses")


class TestDiurnalProfile(unittest.TestCase):
    """The whole point of DiurnalArrivalProfile is that rush hour is
    busier than 3am and weekends are quieter than weekdays -- if either
    of those flips, every downstream conclusion in the README is wrong."""

    def setUp(self):
        self.profile = DiurnalArrivalProfile(base_rates={"N": 0.2, "S": 0.2, "E": 0.1, "W": 0.1})

    def test_morning_rush_busier_than_night(self):
        night = self.profile.rate_at(t_abs=2 * 3600)       # 2am, day 0 (Monday)
        rush = self.profile.rate_at(t_abs=8 * 3600)        # 8am, day 0
        self.assertGreater(rush["N"], night["N"])

    def test_evening_rush_busier_than_midday(self):
        midday = self.profile.rate_at(t_abs=13 * 3600)
        evening = self.profile.rate_at(t_abs=17 * 3600)
        self.assertGreater(evening["N"], midday["N"])

    def test_weekend_quieter_than_weekday_at_same_hour(self):
        weekday_rush = self.profile.rate_at(t_abs=8 * 3600)                   # Monday 8am (day 0)
        weekend_same_hour = self.profile.rate_at(t_abs=5 * 86400 + 8 * 3600)  # Saturday 8am (day 5)
        self.assertLess(weekend_same_hour["N"], weekday_rush["N"])


class TestActionEncoding(unittest.TestCase):
    """decode_joint_action must round-trip: every joint action integer
    used by DQN has to decode to the exact per-node bits it was built
    from, or DQN is silently training against the wrong action space."""

    def test_round_trip_all_combinations(self):
        n_nodes = 4
        for a_idx in range(2 ** n_nodes):
            bits = decode_joint_action(a_idx, n_nodes)
            self.assertEqual(len(bits), n_nodes)
            reconstructed = sum(b << i for i, b in enumerate(bits))
            self.assertEqual(reconstructed, a_idx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
