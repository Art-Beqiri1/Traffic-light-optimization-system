"""
FlowMind — Multi-Day Algorithm Comparison
=============================================

Same five controllers as train_compare.py, but:

  1. Traffic now follows a realistic 24-hour, day/week curve
     (DiurnalArrivalProfile): quiet overnight, morning rush, midday
     plateau, heavier evening rush, flattened weekends -- driven by
     ABSOLUTE simulated time, so "hour 8" always means the morning rush
     no matter how long the run is.

  2. Training episodes are still short (a window of --train-window
     seconds), but each one starts at a RANDOM point across
     --train-span-days days, so agents see quiet nights, both rush
     hours, and weekends during training rather than just one bulge.

  3. EVALUATION runs continuously for --eval-days full days (default 3),
     starting at hour 0 of day 0, so you can see how each controller
     holds up across repeated rush hours over a sustained period --
     not just a single 20-30 minute snapshot.

Usage:
    python train_compare_longrun.py                          # 3-day eval, quick training
    python train_compare_longrun.py --eval-days 7             # a full week
    python train_compare_longrun.py --skip ppo marl           # faster, fewer controllers
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

from network_env import TrafficNetwork, DiurnalArrivalProfile
from controllers import FixedTimerPolicy, MaxPressurePolicy, run_episode
from agents import (
    train_dqn, DQNPolicy,
    train_ppo_centralized, PPOPolicy,
    train_multi_agent_ppo, MultiAgentPPOPolicy,
)

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

PROFILE = DiurnalArrivalProfile(base_rates={"N": 0.22, "S": 0.19, "E": 0.11, "W": 0.09})


def pct_change(base, new):
    return 0.0 if base == 0 else (new - base) / base * 100.0


def print_table(summaries):
    header = f"{'Metric':<26}" + "".join(f"{s['controller'][:18]:>20}" for s in summaries)
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
        vals = "".join(f"{s[key]:>20}" for s in summaries)
        print(f"{label:<26}{vals}")
    print("-" * len(header))


def hourly_profile(queue_series, seconds_per_bucket=3600):
    """Average queue length bucketed by absolute hour, for the 'shape
    over the day(s)' chart."""
    arr = np.array(queue_series, dtype=float)
    n_buckets = int(np.ceil(len(arr) / seconds_per_bucket))
    return np.array([arr[i * seconds_per_bucket:(i + 1) * seconds_per_bucket].mean()
                      for i in range(n_buckets)])


def hour_of_day_profile(queue_series):
    """Average queue length by hour-of-day (0-23), averaged across
    however many days the run covers -- shows the daily rhythm each
    controller settles into once rush hours repeat."""
    arr = np.array(queue_series, dtype=float)
    hours = (np.arange(len(arr)) // 3600) % 24
    return np.array([arr[hours == h].mean() if np.any(hours == h) else 0.0 for h in range(24)])


def plot_multiday_timeline(summaries, path, total_seconds):
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)
    colors = ["#c9463d", "#e08a2b", "#3b6fb6", "#1f8a5f", "#7a4fc9"]
    n_days = int(np.ceil(total_seconds / 86400))

    # shade rush-hour windows each day so the repeating structure is visible
    for d in range(n_days):
        for start_h, end_h in [(7, 9.5), (16, 19)]:
            ax.axvspan(d * 24 + start_h, d * 24 + end_h, color="#888888", alpha=0.08, lw=0)
    for d in range(1, n_days):
        ax.axvline(d * 24, color="#999999", lw=0.8, linestyle="--", alpha=0.6)

    hours_axis = np.arange(total_seconds) / 3600.0
    for s, color in zip(summaries, colors):
        buckets = hourly_profile(s["queue_series"], seconds_per_bucket=300)  # 5-min buckets
        bucket_hours = np.arange(len(buckets)) * (300 / 3600.0)
        ax.plot(bucket_hours, buckets, color=color, lw=1.6, label=s["controller"])

    ax.set_xlabel("Time (hours) — shaded bands = morning/evening rush, dashed lines = day boundaries")
    ax.set_ylabel("Total network queue (5-min avg)")
    ax.set_title(f"Network-wide queue over {n_days} day(s) — same traffic, five controllers")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hour_of_day(summaries, path):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    colors = ["#c9463d", "#e08a2b", "#3b6fb6", "#1f8a5f", "#7a4fc9"]
    for s, color in zip(summaries, colors):
        prof = hour_of_day_profile(s["queue_series"])
        ax.plot(range(24), prof, color=color, lw=2, marker="o", markersize=3, label=s["controller"])
    ax.axvspan(7, 9.5, color="#888888", alpha=0.1, lw=0)
    ax.axvspan(16, 19, color="#888888", alpha=0.1, lw=0)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (averaged across all evaluated days)")
    ax.set_ylabel("Avg total network queue")
    ax.set_title("Daily rhythm each controller settles into (shaded = rush hours)")
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
    fig.suptitle("All five controllers — multi-day evaluation, identical traffic", y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--eval-days", type=float, default=3, help="continuous evaluation length, in days")
    ap.add_argument("--train-window", type=int, default=900, help="seconds per training episode")
    ap.add_argument("--train-span-days", type=int, default=7,
                     help="training episodes start at a random point across this many days")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--green", type=int, default=25, help="fixed-timer green seconds")
    ap.add_argument("--dqn-episodes", type=int, default=150)
    ap.add_argument("--ppo-episodes", type=int, default=180)
    ap.add_argument("--marl-episodes", type=int, default=150)
    ap.add_argument("--skip", nargs="*", default=[], choices=["dqn", "ppo", "marl"])
    args = ap.parse_args()

    # See train_compare.py for why this matters: without it, weight init,
    # PPO action sampling, and epsilon-greedy exploration are unseeded even
    # though the traffic itself and each episode's start_time are already
    # deterministic per-seed below.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    node_ids = [(r, c) for r in range(args.rows) for c in range(args.cols)]
    n_nodes = len(node_ids)
    eval_seconds = int(args.eval_days * 86400)
    train_span_seconds = args.train_span_days * 86400

    def train_factory(seed):
        start = random.Random(seed).randrange(0, train_span_seconds)
        return TrafficNetwork(args.rows, args.cols, PROFILE, args.train_window, seed, start_time=start)

    def eval_factory(seed):
        return TrafficNetwork(args.rows, args.cols, PROFILE, eval_seconds, seed, start_time=0)

    print("=" * 70)
    print(f" FlowMind — {args.rows}x{args.cols} network, {args.eval_days:g}-day evaluation")
    print(f" (training on {args.train_window}s windows sampled across {args.train_span_days} days)")
    print("=" * 70)

    summaries = []

    print("Running Fixed-timer baseline over the full evaluation window...")
    t0 = time.time()
    summaries.append(run_episode(lambda: eval_factory(args.seed), FixedTimerPolicy(args.green), eval_seconds))
    print(f"  done in {time.time()-t0:.1f}s")

    print("Running Max-Pressure over the full evaluation window...")
    t0 = time.time()
    summaries.append(run_episode(lambda: eval_factory(args.seed), MaxPressurePolicy(), eval_seconds))
    print(f"  done in {time.time()-t0:.1f}s")

    if "dqn" not in args.skip:
        print(f"Training Double DQN ({args.dqn_episodes} episodes x {args.train_window}s, "
              f"random start times across {args.train_span_days} days)...")
        t0 = time.time()
        q_net = train_dqn(train_factory, n_nodes, node_ids, args.train_window, args.dqn_episodes)
        print(f"  trained in {time.time()-t0:.1f}s")
        print("Evaluating Double DQN over the full evaluation window...")
        t0 = time.time()
        summaries.append(run_episode(lambda: eval_factory(args.seed), DQNPolicy(q_net, node_ids), eval_seconds))
        print(f"  done in {time.time()-t0:.1f}s")

    if "ppo" not in args.skip:
        print(f"Training PPO ({args.ppo_episodes} episodes x {args.train_window}s)...")
        t0 = time.time()
        ppo_model = train_ppo_centralized(train_factory, n_nodes, node_ids, args.train_window, args.ppo_episodes)
        print(f"  trained in {time.time()-t0:.1f}s")
        print("Evaluating PPO over the full evaluation window...")
        t0 = time.time()
        summaries.append(run_episode(lambda: eval_factory(args.seed), PPOPolicy(ppo_model, node_ids), eval_seconds))
        print(f"  done in {time.time()-t0:.1f}s")

    if "marl" not in args.skip:
        print(f"Training Multi-Agent PPO ({args.marl_episodes} episodes x {args.train_window}s)...")
        t0 = time.time()
        marl_model = train_multi_agent_ppo(train_factory, n_nodes, node_ids, args.train_window, args.marl_episodes)
        print(f"  trained in {time.time()-t0:.1f}s")
        print("Evaluating Multi-Agent PPO over the full evaluation window...")
        t0 = time.time()
        summaries.append(run_episode(lambda: eval_factory(args.seed), MultiAgentPPOPolicy(marl_model, node_ids), eval_seconds))
        print(f"  done in {time.time()-t0:.1f}s")

    print("-" * 70)
    print_table(summaries)

    slim = [{k: v for k, v in s.items() if k != "queue_series"} for s in summaries]
    with open(os.path.join(OUT, "longrun_results.json"), "w") as f:
        json.dump({"config": vars(args), "eval_seconds": eval_seconds, "results": slim}, f, indent=2)

    plot_multiday_timeline(summaries, os.path.join(OUT, "longrun_timeline.png"), eval_seconds)
    plot_hour_of_day(summaries, os.path.join(OUT, "longrun_hour_of_day.png"))
    plot_bar_comparison(summaries, os.path.join(OUT, "longrun_metric_comparison.png"))

    print(f"\nSaved: {OUT}/longrun_results.json")
    print(f"Saved: {OUT}/longrun_timeline.png")
    print(f"Saved: {OUT}/longrun_hour_of_day.png")
    print(f"Saved: {OUT}/longrun_metric_comparison.png")


if __name__ == "__main__":
    main()
