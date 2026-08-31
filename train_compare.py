"""
FlowMind — Multi-Intersection Algorithm Comparison
======================================================

Trains/evaluates FIVE controllers on the SAME small grid network and the
SAME held-out evaluation traffic (identical seed => identical vehicles
arriving at identical seconds for every controller), so the comparison
is apples-to-apples:

  1. Fixed-timer      (today's status quo, blind to demand)
  2. Max-Pressure      (classical rule-based traffic-engineering baseline)
  3. DQN                (centralized, joint-action, value-based)
  4. PPO                (centralized, per-node action heads, policy-based)
  5. Multi-Agent PPO    (decentralized — each intersection is its own agent,
                         local observation + local reward only)

Usage:
    python train_compare.py                       # quick settings
    python train_compare.py --duration 1800 --dqn-episodes 200 --ppo-episodes 200

Note on runtime: DQN/PPO/Multi-Agent PPO need real gradient-descent
training (unlike the original tabular Q-learning script, which trains in
seconds). Default settings below are tuned to run in a few minutes on a
laptop CPU; bump up --*-episodes for better-trained policies before a
final report.
"""
import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from network_env import TrafficNetwork, ArrivalProfile
from controllers import FixedTimerPolicy, MaxPressurePolicy, run_episode
from agents import (
    train_dqn, DQNPolicy,
    train_ppo_centralized, PPOPolicy,
    train_multi_agent_ppo, MultiAgentPPOPolicy,
)

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

PROFILE = ArrivalProfile(base_rates={"N": 0.22, "S": 0.19, "E": 0.11, "W": 0.09})


def make_net_factory(rows, cols, duration):
    def factory(seed):
        return TrafficNetwork(rows, cols, PROFILE, duration, seed)
    return factory


def pct_change(base, new):
    if base == 0:
        return 0.0
    return (new - base) / base * 100.0


def print_table(summaries):
    header = f"{'Metric':<26}" + "".join(f"{s['controller'][:16]:>18}" for s in summaries)
    print(header)
    print("-" * len(header))
    rows = [
        ("Avg wait / vehicle (s)", "avg_wait_per_vehicle_s"),
        ("Peak total queue", "max_queue"),
        ("Avg total queue", "avg_queue"),
        ("Vehicles served", "vehicles_served"),
        ("Phase switches (all nodes)", "phase_switches"),
    ]
    for label, key in rows:
        vals = "".join(f"{s[key]:>18}" for s in summaries)
        print(f"{label:<26}{vals}")
    print("-" * len(header))


def plot_queue_over_time(summaries, path, duration):
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    t = np.arange(duration)
    colors = ["#c9463d", "#e08a2b", "#3b6fb6", "#1f8a5f", "#7a4fc9"]

    def smooth(series, w=15):
        arr = np.array(series, dtype=float)
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    for s, color in zip(summaries, colors):
        ax.plot(t, smooth(s["queue_series"]), color=color, lw=2, label=s["controller"])
    ax.set_xlabel("Simulated time (seconds)")
    ax.set_ylabel("Total vehicles queued, network-wide (smoothed)")
    ax.set_title("Network-wide queue over time — same traffic, five controllers")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bar_comparison(summaries, path):
    metrics = [
        ("avg_wait_per_vehicle_s", "Avg wait / vehicle (s)"),
        ("max_queue", "Peak total queue"),
        ("avg_queue", "Avg total queue"),
    ]
    colors = ["#c9463d", "#e08a2b", "#3b6fb6", "#1f8a5f", "#7a4fc9"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    labels = [s["controller"] for s in summaries]
    for ax, (key, title) in zip(axes, metrics):
        vals = [s[key] for s in summaries]
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
        ax.grid(alpha=0.15, axis="y")
    fig.suptitle("All five controllers — key metrics, identical network traffic", y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--duration", type=int, default=1200, help="evaluation run length, seconds")
    ap.add_argument("--train-duration", type=int, default=600, help="episode length during training, seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--green", type=int, default=25, help="fixed-timer green seconds")
    ap.add_argument("--dqn-episodes", type=int, default=120)
    ap.add_argument("--ppo-episodes", type=int, default=150)
    ap.add_argument("--marl-episodes", type=int, default=120)
    ap.add_argument("--skip", nargs="*", default=[], choices=["dqn", "ppo", "marl"],
                     help="skip one or more learned controllers, e.g. --skip marl")
    args = ap.parse_args()

    # Seed everything from one place. Without this, DQN/PPO/MARL weight
    # init, PPO's action sampling, and epsilon-greedy exploration all draw
    # from unseeded global RNGs -- meaning a "rerun this and check" request
    # (a completely reasonable code-review ask) would not reproduce the
    # numbers in this README. Network traffic itself is already seeded
    # per-call (see eval_factory_seeded/train_factory), so this closes the
    # last gap.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_factory = make_net_factory(args.rows, args.cols, args.train_duration)
    eval_factory_seeded = lambda seed: TrafficNetwork(args.rows, args.cols, PROFILE, args.duration, seed)
    node_ids = [(r, c) for r in range(args.rows) for c in range(args.cols)]
    n_nodes = len(node_ids)

    print("=" * 70)
    print(f" FlowMind — {args.rows}x{args.cols} network ({n_nodes} intersections), 5-way comparison")
    print("=" * 70)

    summaries = []

    print("Running Fixed-timer baseline...")
    summaries.append(run_episode(lambda: eval_factory_seeded(args.seed), FixedTimerPolicy(args.green), args.duration))

    print("Running Max-Pressure...")
    summaries.append(run_episode(lambda: eval_factory_seeded(args.seed), MaxPressurePolicy(), args.duration))

    if "dqn" not in args.skip:
        print(f"Training DQN ({args.dqn_episodes} episodes x {args.train_duration}s)...")
        t0 = time.time()
        q_net = train_dqn(lambda seed: TrafficNetwork(args.rows, args.cols, PROFILE, args.train_duration, seed),
                           n_nodes, node_ids, args.train_duration, args.dqn_episodes)
        print(f"  done in {time.time()-t0:.1f}s")
        print("Evaluating DQN...")
        summaries.append(run_episode(lambda: eval_factory_seeded(args.seed), DQNPolicy(q_net, node_ids), args.duration))

    if "ppo" not in args.skip:
        print(f"Training PPO ({args.ppo_episodes} episodes x {args.train_duration}s)...")
        t0 = time.time()
        ppo_model = train_ppo_centralized(
            lambda seed: TrafficNetwork(args.rows, args.cols, PROFILE, args.train_duration, seed),
            n_nodes, node_ids, args.train_duration, args.ppo_episodes)
        print(f"  done in {time.time()-t0:.1f}s")
        print("Evaluating PPO...")
        summaries.append(run_episode(lambda: eval_factory_seeded(args.seed), PPOPolicy(ppo_model, node_ids), args.duration))

    if "marl" not in args.skip:
        print(f"Training Multi-Agent PPO ({args.marl_episodes} episodes x {args.train_duration}s)...")
        t0 = time.time()
        marl_model = train_multi_agent_ppo(
            lambda seed: TrafficNetwork(args.rows, args.cols, PROFILE, args.train_duration, seed),
            n_nodes, node_ids, args.train_duration, args.marl_episodes)
        print(f"  done in {time.time()-t0:.1f}s")
        print("Evaluating Multi-Agent PPO...")
        summaries.append(run_episode(lambda: eval_factory_seeded(args.seed), MultiAgentPPOPolicy(marl_model, node_ids), args.duration))

    print("-" * 70)
    print_table(summaries)

    # ---- save artifacts ----
    slim = [{k: v for k, v in s.items() if k != "queue_series"} for s in summaries]
    with open(os.path.join(OUT, "network_results.json"), "w") as f:
        json.dump({"config": vars(args), "results": slim}, f, indent=2)

    plot_queue_over_time(summaries, os.path.join(OUT, "network_queue_over_time.png"), args.duration)
    plot_bar_comparison(summaries, os.path.join(OUT, "network_metric_comparison.png"))

    print(f"\nSaved: {OUT}/network_results.json")
    print(f"Saved: {OUT}/network_queue_over_time.png")
    print(f"Saved: {OUT}/network_metric_comparison.png")


if __name__ == "__main__":
    main()
