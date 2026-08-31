"""
Learning agents for the multi-intersection network.

  DQN               : one centralized agent. Sees the WHOLE network's
                       state (all nodes concatenated) and picks a single
                       JOINT action out of 2^N combinations (bit i =
                       keep/switch for node i). Value-based, off-policy,
                       replay buffer + target network.

  PPO               : also centralized (sees the whole network, single
                       shared critic) but policy-based: the actor has
                       one independent 2-way (keep/switch) categorical
                       head per node, trained jointly with PPO-clip +
                       GAE. Scales to more nodes better than DQN's
                       2^N action blow-up.

  MultiAgentPPO     : DECENTRALIZED. Each node is its own agent, sees
                       only ITS OWN local state (no neighbor info) and
                       gets ITS OWN local reward (its own queue, not the
                       network's). Parameters are shared across agents
                       for sample efficiency (a common, standard choice
                       -- "shared-parameter independent PPO" / IPPO) but
                       each agent still acts on local info only at
                       execution time, which is the actual multi-agent
                       property being demonstrated (no communication
                       between intersections, unlike the centralized
                       PPO above which effectively coordinates because
                       one brain sees everything).
"""
from __future__ import annotations
import random
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network_env import TrafficNetwork

DEVICE = torch.device("cpu")

# Raw per-tick reward is -(total queue) [+ a -6 switch penalty], which can
# run into the tens/hundreds for a multi-node network. Scaling it down
# keeps value-function targets and policy gradients numerically stable
# without changing what the optimal policy is (scaling a reward signal by
# a positive constant doesn't change the argmax policy).
REWARD_SCALE = 0.05


# ----------------------------------------------------------------------
# DQN (centralized, joint action space)
# ----------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


def decode_joint_action(a: int, n_nodes: int) -> List[int]:
    return [(a >> i) & 1 for i in range(n_nodes)]


def train_dqn(net_factory, n_nodes: int, node_ids, duration: int, episodes: int,
              batch_size=64, update_every=4, target_sync=500,
              lr=1e-3, gamma=0.97, verbose=True) -> QNet:
    state_dim = n_nodes * 5
    n_actions = 2 ** n_nodes
    q = QNet(state_dim, n_actions).to(DEVICE)
    q_target = QNet(state_dim, n_actions).to(DEVICE)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer()

    step_count = 0
    for ep in range(episodes):
        epsilon = max(0.05, 0.9 * (1 - ep / episodes))
        net: TrafficNetwork = net_factory(seed=2000 + ep)
        state = net.reset()
        for t in range(duration):
            if random.random() < epsilon:
                a_idx = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    qv = q(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    a_idx = int(qv.argmax(dim=1).item())
            bits = decode_joint_action(a_idx, n_nodes)
            actions = {rc: bits[i] for i, rc in enumerate(node_ids)}
            next_state, rewards, done, _ = net.step(actions)
            reward = sum(rewards.values()) * REWARD_SCALE
            buf.push(state, a_idx, reward, next_state, float(done))
            state = next_state
            step_count += 1

            if len(buf) >= batch_size and step_count % update_every == 0:
                s, a, r, s2, d = buf.sample(batch_size)
                qsa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # Double DQN: pick the best next action with the ONLINE
                    # network, but evaluate its value with the TARGET
                    # network. Plain DQN uses the same network for both,
                    # which systematically overestimates Q-values (it's
                    # biased toward whichever action currently looks best,
                    # including by noise) -- decoupling selection from
                    # evaluation is a well-established, nearly-free fix.
                    best_next_action = q(s2).argmax(dim=1, keepdim=True)
                    max_q_next = q_target(s2).gather(1, best_next_action).squeeze(1)
                    target = r + gamma * max_q_next * (1 - d)
                loss = F.smooth_l1_loss(qsa, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

            if step_count % target_sync == 0:
                q_target.load_state_dict(q.state_dict())

        if verbose and (ep + 1) % max(1, episodes // 10) == 0:
            print(f"  DQN episode {ep+1}/{episodes}  epsilon={epsilon:.2f}")

    return q


class DQNPolicy:
    name = "Double DQN (centralized)"

    def __init__(self, q_net: QNet, node_ids):
        self.q = q_net
        self.node_ids = node_ids
        self.n_nodes = len(node_ids)

    def act(self, net: TrafficNetwork) -> dict:
        state = net._global_state()
        with torch.no_grad():
            qv = self.q(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            a_idx = int(qv.argmax(dim=1).item())
        bits = decode_joint_action(a_idx, self.n_nodes)
        return {rc: bits[i] for i, rc in enumerate(self.node_ids)}


# ----------------------------------------------------------------------
# PPO (centralized, per-node categorical heads, shared critic)
# ----------------------------------------------------------------------
class ActorCriticCentralized(nn.Module):
    def __init__(self, state_dim, n_nodes, hidden=128):
        super().__init__()
        self.n_nodes = n_nodes
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor_heads = nn.Linear(hidden, n_nodes * 2)   # 2 logits per node
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        logits = self.actor_heads(h).view(-1, self.n_nodes, 2)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def act(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()                       # (batch, n_nodes)
        logp = dist.log_prob(action).sum(dim=-1)      # joint log-prob
        return action, logp, value

    def evaluate(self, state, action):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy, value


def _gae(rewards, values, dones, gamma=0.97, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_adv = delta + gamma * lam * next_nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + np.array(values[:T], dtype=np.float32)
    return advantages, returns


def train_ppo_centralized(net_factory, n_nodes, node_ids, duration, total_episodes,
                           rollout_len=2048, epochs=4, minibatch=256,
                           lr=1e-4, gamma=0.97, clip=0.2, verbose=True) -> ActorCriticCentralized:
    state_dim = n_nodes * 5
    model = ActorCriticCentralized(state_dim, n_nodes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ep = 0
    net = net_factory(seed=3000)
    state = net.reset()
    t_in_ep = 0

    while ep < total_episodes:
        states, actions, logps, values, rewards, dones = [], [], [], [], [], []
        for _ in range(rollout_len):
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, logp, value = model.act(st)
            bits = action.squeeze(0).tolist()
            act_dict = {rc: bits[i] for i, rc in enumerate(node_ids)}
            next_state, node_rewards, done, _ = net.step(act_dict)
            reward = sum(node_rewards.values()) * REWARD_SCALE

            states.append(state)
            actions.append(bits)
            logps.append(logp.item())
            values.append(value.item())
            rewards.append(reward)
            dones.append(float(done))

            state = next_state
            t_in_ep += 1
            if done or t_in_ep >= duration:
                ep += 1
                t_in_ep = 0
                net = net_factory(seed=3000 + ep)
                state = net.reset()
                if ep >= total_episodes:
                    break

        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            _, _, last_value = model.act(st)
        values_ext = values + [last_value.item()]
        advantages, returns = _gae(rewards, values_ext, dones, gamma=gamma)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long)
        old_logp_t = torch.tensor(logps, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        n = len(states)
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, minibatch):
                mb = idx[start:start + minibatch]
                mb = torch.tensor(mb, dtype=torch.long)
                logp_new, entropy, value_new = model.evaluate(states_t[mb], actions_t[mb])
                ratio = torch.exp(logp_new - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value_new, ret_t[mb])
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        if verbose and ep % max(1, total_episodes // 10) == 0:
            print(f"  PPO episode {ep}/{total_episodes}  mean rollout reward={np.mean(rewards):.2f}")

    return model


class PPOPolicy:
    name = "PPO (centralized)"

    def __init__(self, model: ActorCriticCentralized, node_ids):
        self.model = model
        self.node_ids = node_ids

    def act(self, net: TrafficNetwork) -> dict:
        state = net._global_state()
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(st)
            bits = logits.argmax(dim=-1).squeeze(0).tolist()  # greedy eval
        return {rc: bits[i] for i, rc in enumerate(self.node_ids)}


# ----------------------------------------------------------------------
# Multi-agent PPO (decentralized, shared weights, local obs + local reward)
# ----------------------------------------------------------------------
class ActorCriticLocal(nn.Module):
    """One small actor-critic, applied independently per node (shared
    weights across nodes = parameter sharing, but each forward pass only
    ever sees ONE node's own 5 local features — no cross-node info)."""

    def __init__(self, local_dim=5, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(local_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, 2)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, x):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def evaluate(self, x, action):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        return logp, entropy, value


def train_multi_agent_ppo(net_factory, n_nodes, node_ids, duration, total_episodes,
                           rollout_len=2048, epochs=4, minibatch=256,
                           lr=1e-4, gamma=0.97, clip=0.2, verbose=True) -> ActorCriticLocal:
    model = ActorCriticLocal().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ep = 0
    net = net_factory(seed=4000)
    net.reset()
    t_in_ep = 0

    while ep < total_episodes:
        # buffers per node (each node = one independent trajectory, but
        # all trajectories are pooled into one big batch to update the
        # single shared network)
        buf = {rc: {"s": [], "a": [], "logp": [], "v": [], "r": [], "d": []} for rc in node_ids}

        for _ in range(rollout_len):
            local_states = {rc: net.local_features(rc) for rc in node_ids}
            batch_local = torch.tensor([local_states[rc] for rc in node_ids], dtype=torch.float32)
            with torch.no_grad():
                actions_t, logps_t, values_t = model.act(batch_local)
            act_dict = {}
            for i, rc in enumerate(node_ids):
                act_dict[rc] = int(actions_t[i].item())
                buf[rc]["s"].append(local_states[rc])
                buf[rc]["a"].append(int(actions_t[i].item()))
                buf[rc]["logp"].append(logps_t[i].item())
                buf[rc]["v"].append(values_t[i].item())

            _, node_rewards, done, _ = net.step(act_dict)
            for rc in node_ids:
                buf[rc]["r"].append(node_rewards[rc] * REWARD_SCALE)
                buf[rc]["d"].append(float(done))

            t_in_ep += 1
            if done or t_in_ep >= duration:
                ep += 1
                t_in_ep = 0
                net = net_factory(seed=4000 + ep)
                net.reset()
                if ep >= total_episodes:
                    break

        # bootstrap value for the tail of each node's trajectory
        local_states = {rc: net.local_features(rc) for rc in node_ids}
        batch_local = torch.tensor([local_states[rc] for rc in node_ids], dtype=torch.float32)
        with torch.no_grad():
            _, last_values = model.forward(batch_local)

        all_states, all_actions, all_logp, all_adv, all_ret = [], [], [], [], []
        for i, rc in enumerate(node_ids):
            values_ext = buf[rc]["v"] + [last_values[i].item()]
            adv, ret = _gae(buf[rc]["r"], values_ext, buf[rc]["d"], gamma=gamma)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            all_states.extend(buf[rc]["s"])
            all_actions.extend(buf[rc]["a"])
            all_logp.extend(buf[rc]["logp"])
            all_adv.extend(adv.tolist())
            all_ret.extend(ret.tolist())

        states_t = torch.tensor(np.array(all_states), dtype=torch.float32)
        actions_t = torch.tensor(all_actions, dtype=torch.long)
        old_logp_t = torch.tensor(all_logp, dtype=torch.float32)
        adv_t = torch.tensor(all_adv, dtype=torch.float32)
        ret_t = torch.tensor(all_ret, dtype=torch.float32)

        n = len(all_states)
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, minibatch):
                mb = torch.tensor(idx[start:start + minibatch], dtype=torch.long)
                logp_new, entropy, value_new = model.evaluate(states_t[mb], actions_t[mb])
                ratio = torch.exp(logp_new - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value_new, ret_t[mb])
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        if verbose and ep % max(1, total_episodes // 10) == 0:
            print(f"  Multi-agent PPO episode {ep}/{total_episodes}")

    return model


class MultiAgentPPOPolicy:
    name = "Multi-Agent PPO (decentralized)"

    def __init__(self, model: ActorCriticLocal, node_ids):
        self.model = model
        self.node_ids = node_ids

    def act(self, net: TrafficNetwork) -> dict:
        local_states = {rc: net.local_features(rc) for rc in self.node_ids}
        batch_local = torch.tensor([local_states[rc] for rc in self.node_ids], dtype=torch.float32)
        with torch.no_grad():
            logits, _ = self.model(batch_local)
            bits = logits.argmax(dim=-1).tolist()
        return {rc: bits[i] for i, rc in enumerate(self.node_ids)}
