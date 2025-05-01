# dqn_task3.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb

gym.register_envs(ale_py)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=1000000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.frame = 1

    def beta_by_frame(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, transition, error):
        p = (abs(error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame()
        self.frame += 1
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return np.stack(states), actions, rewards, np.stack(next_states), dones, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

class MultiStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def push(self, transition):
        self.buffer.append(transition)

    def is_ready(self):
        return len(self.buffer) == self.buffer.maxlen

    def get(self):
        R, (s, a, _, _, _) = 0, self.buffer[0]
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
            if d:
                break
        s_next, _, _, s_last, d_last = self.buffer[-1]
        return (s, a, R, s_last, d_last)

class DQNAgent:
    def __init__(self, args):
        self.env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        self.test_env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        self.preprocessor = AtariPreprocessor()
        self.num_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100000, gamma=0.5)


        self.memory = PrioritizedReplayBuffer(args.memory_size)
        self.multi_step = MultiStepBuffer(args.n_step, args.gamma)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.target_update = args.target_update
        self.train_start = args.train_start
        self.train_freq = args.train_freq

        self.env_step = 0
        self.best_reward = -21
        self.save_path = args.save_path
        os.makedirs(self.save_path, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def train(self):
        if len(self.memory.buffer) < self.train_start:
            return
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)
        weights = torch.tensor(weights).float().to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma ** args.n_step * (1 - dones) * next_q_values

        loss = (q_values - targets).pow(2) * weights
        prios = loss.detach().cpu().numpy() + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
        total_norm = 0
        for p in self.q_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.memory.update_priorities(indices, prios)
        td_errors = (q_values - targets).detach()
        if self.env_step % 100 == 0:
            wandb.log({
                "loss": loss.item(),
                "avg_q_value": q_values.mean().item(),
                "td_error_mean": td_errors.abs().mean().item(),
                "grad_norm": total_norm,
                "env_step": self.env_step,
                "lr": self.optimizer.param_groups[0]['lr']
            })

        if self.env_step % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


    def run(self, episodes):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            while not done:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocessor.step(next_obs)

                self.multi_step.push((state, action, reward, next_state, done))
                if self.multi_step.is_ready():
                    transition = self.multi_step.get()
                    with torch.no_grad():
                        s, a, R, s_, d = transition
                        s_t = torch.from_numpy(np.array(s)).unsqueeze(0).float().to(self.device)
                        q = self.q_net(s_t)
                        error = R - q[0][a].item()
                    self.memory.add(transition, error)

                state = next_state
                total_reward += reward
                self.env_step += 1

                if self.env_step % self.train_freq == 0:
                    self.train()
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                if self.env_step % 200_000 == 0:
                    torch.save(self.q_net.state_dict(), os.path.join(self.save_path, f"LAB5_313552013_task3_pong{self.env_step}.pt"))

            wandb.log({"episode": ep, "reward": total_reward, "epsilon": self.epsilon, "env_step": self.env_step, "lr": self.optimizer.param_groups[0]['lr']})
            print(f"[train] Ep: {ep}, Reward: {total_reward}, Epsilon: {self.epsilon:.4f}, env_step: {self.env_step}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            if ep > 5 and ep % 20 == 0:
                avg_reward = self.evaluate()
                print(f"Ep {ep}, Avg Eval Reward: {avg_reward:.2f}")
                wandb.log({"eval_reward": avg_reward, "env_step": self.env_step})
                if avg_reward > 0 and avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    torch.save(self.q_net.state_dict(), os.path.join(self.save_path, f"best_model_env_step_{self.env_step}_reward_{self.best_reward:.2f}.pt"))

    def evaluate(self, num_episodes=20):
        total_reward = 0
        for i in range(num_episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            episode_reward = 0
            while not done:
                state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                state = self.preprocessor.step(next_obs)
                total_reward += reward
                episode_reward += reward
            print(f"[{i}] Eval Reward: {episode_reward}")
            
        avg_reward = total_reward / num_episodes
        return avg_reward

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-decay', type=float, default=0.9995)
    parser.add_argument('--epsilon-min', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--memory-size', type=int, default=100000)
    parser.add_argument('--target-update', type=int, default=1000)
    parser.add_argument('--train-start', type=int, default=10000)
    parser.add_argument('--train-freq', type=int, default=4)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--save-path', type=str, default="./dqn_task3_models")
    parser.add_argument('--wandb-run-name', type=str, default="pong-enhanced-dqn")
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-Task3", name=args.wandb_run_name)
    agent = DQNAgent(args)
    agent.run(args.episodes)
