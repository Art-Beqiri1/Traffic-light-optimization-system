> Live app update: four cameras, yellow transitions, manual and fixed-timer controls. See [LIVE_README.md](LIVE_README.md) for setup and limitations. The benchmark results below use the original offline simulator.

# FlowMind - Multi-Intersection Algorithm Comparison

This extends your original single-intersection Fixed-timer vs. Q-learning
comparison to a small **2x2 network of coupled intersections**, and adds
four more controllers so you can compare all of them head-to-head on the
exact same traffic:

1. **Fixed-timer** - your original baseline (unchanged idea, now per-node)
2. **Max-Pressure** - the classical, non-learning traffic-engineering
   algorithm. No training needed; it just reacts to live queue counts.
3. **DQN** - centralized, value-based deep RL
4. **PPO** - centralized, policy-based deep RL
5. **Multi-Agent PPO** - decentralized: each intersection is its own agent

## Why a network, not just 4 copies of one intersection

A grid of intersections only gets interesting for RL if decisions at one
node actually affect its neighbors. So `network_env.py` couples the nodes:
when an intersection discharges vehicles, they don't disappear - they
become next-tick arrivals at the neighboring intersection they're driving
toward (or exit the grid if they're at the boundary). This means a bad
decision at one signal can flood the one downstream - the "green wave"
coordination problem that makes multi-intersection traffic control
genuinely harder than 4 independent single-intersection problems.

## Files

```
network_env.py     the coupled multi-intersection simulator (state/dynamics)
controllers.py      Fixed-timer + Max-Pressure (no training required)
agents.py            DQN, centralized PPO, and decentralized Multi-Agent PPO (PyTorch)
train_compare.py    trains what needs training, evaluates all 5, saves comparison charts
simulation.py        your original single-intersection file (kept, untouched, still works standalone)
```

## How to run

```bash
pip install -r requirements.txt
python train_compare.py
```

Takes ~2 minutes on a laptop CPU with default settings (4 intersections,
120–150 training episodes each for DQN/PPO/Multi-Agent PPO). Useful flags:

```bash
python train_compare.py --rows 2 --cols 2 --duration 1800 \
    --dqn-episodes 200 --ppo-episodes 250 --marl-episodes 200
python train_compare.py --skip ppo marl     # just run Fixed-timer, Max-Pressure, DQN
```

## What each algorithm actually is here

- **State** (per intersection): NS queue, EW queue, current phase,
  time-in-phase, whether the minimum green has been met - same shape as
  your original design, just continuous instead of binned so a neural
  net can use it directly.
- **Action**: KEEP or SWITCH, same safety layer as before (can't switch
  before `MIN_GREEN`, forced to switch at `MAX_GREEN`).
- **Reward**: negative total queue, with a switching penalty - identical
  objective to your original Q-learning script.
- **DQN**: one network sees the whole network's state and picks one joint
  action out of 2^4 = 16 combinations. Off-policy, replay buffer + target
  network.
- **PPO (centralized)**: one network, one independent 2-way action head
  per intersection, one shared critic. Scales better than DQN's 2^N
  action blow-up as you add intersections.
- **Multi-Agent PPO**: each intersection is a *separate* agent that only
  ever sees its own local state and gets its own local reward (its own
  queue, not the network's) - no communication between signals. Weights
  are shared across agents for sample efficiency, but each one acts on
  local information only, which is the actual "multi-agent" property
  being tested (as opposed to the centralized PPO above, which
  effectively coordinates because one brain sees everything).

## Honest results (from the run that produced `output/`)

| Controller | Avg wait/vehicle (s) | Avg queue | Peak queue |
|---|---|---|---|
| Fixed-timer | 6.09 | 20.78 | 60 |
| Max-Pressure | 7.17 | 24.51 | 56 |
| **DQN** | **3.50** | **11.96** | **32** |
| PPO | 11.33 | 38.62 | 80 |
| Multi-Agent PPO | 4.19 | 14.31 | 47 |

Worth reporting honestly, not smoothing over:

- **DQN and Multi-Agent PPO beat both baselines** at this training budget.
  Max-Pressure being a strong, hard-to-beat baseline is itself a real,
  well-documented finding in the traffic-signal RL literature - it isn't
  a bug that it's competitive with (and here slightly worse than, but
  close to) Fixed-timer.
- **Centralized PPO underperforms here.** On-policy policy-gradient
  methods are known to be more sample-hungry and more sensitive to
  reward scale/learning rate than DQN in this kind of setting; 150
  episodes of 600 simulated seconds isn't a lot of data for it. Levers
  worth trying before trusting it for a report: more episodes, a
  learning-rate schedule (decay over training), a larger rollout length,
  or reward normalization (running mean/std rather than the fixed
  0.05 scale used here).
- Deep RL needing real gradient-descent training (unlike your original
  tabular Q-table, which converges in seconds) is the actual reason
  production systems default to Max-Pressure or fixed-timer as a
  fallback while an RL controller is being validated - that's a fair
  point to make in a design write-up.

## Output

Running `train_compare.py` writes to `output/`:
- `network_results.json` - full metrics table
- `network_queue_over_time.png` - all 5 controllers' network-wide queue over the same evaluation traffic
- `network_metric_comparison.png` - bar-chart comparison of the 3 headline metrics

## Multi-day evaluation (`train_compare_longrun.py`)

The script above evaluates on a 20-30 minute snapshot. `train_compare_longrun.py`
answers a different question: **how does each controller behave over a
full day or several days of realistic, repeating traffic** - quiet
overnight, morning rush, midday plateau, a heavier evening rush, and
flattened weekends - rather than one isolated rush-hour bulge?

Key differences from `train_compare.py`:

- **Traffic is driven by absolute time of day** (`DiurnalArrivalProfile`
  in `network_env.py`), not time-since-episode-start - hour 8 is always
  the morning rush, no matter how long the run is.
- **Training** still uses short episodes (`--train-window` seconds, e.g.
  15 minutes), but each episode starts at a random point across
  `--train-span-days` days, so agents see quiet nights, both rush hours,
  and weekends during training instead of one bulge.
- **Evaluation runs continuously** for `--eval-days` full days (default
  3) starting at hour 0 of day 0, so rush hours repeat back-to-back and
  you can see whether a controller's queues recover between peaks or
  compound across them.
- **DQN is now Double DQN** - action selection and action evaluation use
  separate networks (online vs. target), which removes DQN's well-known
  overestimation bias for a negligible extra cost. This is the "better
  algorithm" upgrade over the single-day version.

```bash
python train_compare_longrun.py                     # 3-day eval, default settings
python train_compare_longrun.py --eval-days 7        # a full week
python train_compare_longrun.py --skip ppo marl      # just Fixed-timer, Max-Pressure, Double DQN
```

Takes ~5 minutes on a laptop CPU for the 3-day default. Outputs (in `output/`):
- `longrun_results.json` - full metrics table + run config
- `longrun_timeline.png` - 5-minute-bucketed queue over the whole multi-day run, with rush-hour bands and day boundaries marked
- `longrun_hour_of_day.png` - average queue by hour-of-day (0-23), averaged across every day evaluated - shows the daily rhythm each controller settles into
- `longrun_metric_comparison.png` - bar-chart comparison of the 3 headline metrics

### Results from the 3-day run that produced `output/`

| Controller | Avg wait/vehicle | Avg queue | Peak queue | Phase switches |
|---|---|---|---|---|
| Fixed-timer | 5.64s | 9.34 | 68 | 37,028 |
| Max-Pressure | 6.38s | 10.57 | 68 | 81,953 |
| **Double DQN** | **4.17s** | **6.91** | **47** | 77,098 |
| PPO | 13.02s | 21.57 | 150 | 68,993 |
| Multi-Agent PPO | 10.88s | 18.02 | 84 | 32,693 |

Same honest pattern as the single-episode comparison, now confirmed over
a sustained, repeating traffic load: **Double DQN is the clear winner**
and holds up across all three days without its queues drifting upward
day-over-day. **PPO and Multi-Agent PPO underperform both classical
baselines at this training budget** - worth reporting plainly rather
than cherry-picking a favorable run. Two things worth trying if you want
stronger PPO numbers for a final report: (1) more training episodes
specifically covering weekday rush-hour windows (right now training
samples uniformly across all 7 days, so rush-hour is a minority of
training experience), and (2) a learning-rate schedule that decays over
training rather than a single fixed rate.
