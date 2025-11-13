"""
Neural Actor–Critic brain with replay buffer for society_sim.py
"""

import random, collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ACTIONS = ["move", "gather", "help", "donate", "trade", "steal", "tag", "idle"]

# -------------------------------
# Neural networks
# -------------------------------

class ActorCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.actor = nn.Linear(32, action_size)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


# -------------------------------
# Replay buffer
# -------------------------------

class ReplayBuffer:
    def __init__(self, capacity=500):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)


# -------------------------------
# Wrapper with training logic
# -------------------------------

class BrainWrapper:
    def __init__(self, state_size: int, lr: float = 1e-3):
        self.device = torch.device("cpu")
        self.model = ActorCritic(state_size, len(ACTIONS)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = ReplayBuffer(500)
        self.gamma = 0.99
        self.last_logprob = None
        self.last_value = None
        self.last_state = None
        self.last_action = None

    def select_action(self, state_tensor):
        logits, value = self.model(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        logprob = dist.log_prob(action_idx)
        self.last_logprob = logprob
        self.last_value = value
        self.last_action = action_idx.item()
        return ACTIONS[action_idx.item()]

    def learn(self, next_state_tensor, reward: float, done: bool):
        if self.last_logprob is None:
            return
        gamma = self.gamma
        _, next_value = self.model(next_state_tensor)
        target_value = reward + (0 if done else gamma * next_value.detach())
        advantage = target_value - self.last_value
        actor_loss = -self.last_logprob * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + 0.5 * critic_loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Store in buffer
        self.buffer.push(self.last_state, self.last_action, reward, next_state_tensor, done)

    def replay_train(self, batch_size=16):
        if len(self.buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        logits, values = self.model(states)
        _, next_values = self.model(next_states)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        chosen_log_probs = log_probs[range(len(actions)), actions]

        targets = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantages = targets.detach() - values.squeeze()

        actor_loss = -(chosen_log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

