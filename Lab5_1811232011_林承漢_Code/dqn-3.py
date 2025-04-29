# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

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
import argparse
import time
import contextlib

gym.register_envs(ale_py)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions, input_shape):
        super(DQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########
        if len(input_shape) == 1:            # CartPole → 向量
            self.network = nn.Sequential(
                nn.Linear(input_shape[0], 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, num_actions)
            )
        else:                                # Pong → 影像 (4,84,84)
            self.network = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),     nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),     nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64*7*7, 512),      nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
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


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
        

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print("Using device:", self.device)

        # Uniform replay buffer for Task 2 (Vanilla DQN on Pong)
        self.memory = deque(maxlen=args.memory_size)

        # Build networks with correct input shape after preprocessing one observation
        dummy_obs, _ = self.env.reset()
        dummy_state = self.preprocessor.reset(dummy_obs)
        input_shape = dummy_state.shape  # (4, 84, 84)
        self.q_net = DQN(self.num_actions, input_shape).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, input_shape).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.ep_count = args.ep_count if args else 0
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initialized for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state) / 255.0).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for _ in range(episodes):
            ep = self.ep_count 
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)
                self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    }, step=self.env_count)
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            }, step=self.env_count)
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                self.save_checkpoint(model_path)
                print(f"[Checkpoint] Saved to ckpt_ep{ep}.pt")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    self.save_checkpoint(model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })
            self.ep_count += 1

    def save_checkpoint(self, path):
        ckpt = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optim": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "env_count": self.env_count,
            "train_count": self.train_count,
            "ep_count": self.ep_count
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path):
        if not path or not os.path.isfile(path):
            print(f"[Checkpoint] {path} not found，跳過載入")
            return
        ckpt = torch.load(path, map_location=self.device)

        # ── 1. 先載入 q_net ──
        if isinstance(ckpt, dict) and "q_net" in ckpt:     # 完整 ckpt
            self.q_net.load_state_dict(ckpt["q_net"])
        else:                                              # 只有 q_net weights
            self.q_net.load_state_dict(ckpt)

        # ── 2. target_net：若沒存就複製 q_net 的權重 ──
        target_sd = ckpt.get("target_net") if isinstance(ckpt, dict) else None
        if target_sd is None:
            target_sd = self.q_net.state_dict()
        self.target_net.load_state_dict(target_sd)

        # ── 3. 其餘資訊 ──
        if isinstance(ckpt, dict) and "optim" in ckpt:
            self.optimizer.load_state_dict(ckpt["optim"])
            self.optimizer_to_device()          # 如果用了 MPS/CUDA，確保 tensor 在對的 device
        self.ep_count    = ckpt.get("ep_count",    self.ep_count)
        self.epsilon     = ckpt.get("epsilon",     self.epsilon)     if isinstance(ckpt, dict) else self.epsilon
        self.env_count   = ckpt.get("env_count",   self.env_count)   if isinstance(ckpt, dict) else self.env_count
        self.train_count = ckpt.get("train_count", self.train_count) if isinstance(ckpt, dict) else self.train_count

        print(f"[Checkpoint] Loaded {path}  ε={self.epsilon:.3f}  env={self.env_count}  upd={self.train_count}")

    def optimizer_to_device(self):
        # 將 optimizer 內的 state tensor 搬到 self.device，避免 MPS/CUDA 報錯
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state) / 255.0).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        # Sample a mini‑batch of (s,a,r,s',done) from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        states = torch.from_numpy(states.astype(np.float32) / 255.0).to(self.device)
        next_states = torch.from_numpy(next_states.astype(np.float32) / 255.0).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Use mixed precision on Apple‑Silicon / CUDA
        use_autocast = self.device.type in ['cuda', 'mps']
        autocast_ctx = torch.autocast(device_type=self.device.type, dtype=torch.float16) if use_autocast else contextlib.nullcontext()
        with autocast_ctx:
            q_pred = self.q_net(states)
            q_values = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

            loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="pong-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--load-ckpt", type=str, default="", help="Path to checkpoint")
    parser.add_argument("--ep-count", type=int, default=0, help="Episode count for loading checkpoint")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")
    parser.add_argument("--wandb-id",  type=str, default=None, help="run_id to resume")
    args = parser.parse_args()

    wandb.init(
        project=f"DLP-Lab5-DQN-Pong",   
        name=args.wandb_run_name,
        id=args.wandb_id,  # ← 這就是 run-id
        resume="must" if args.wandb_id is not None else None,     # 必須找到同一條 run，否則報錯
        save_code=True,
    )

    agent = DQNAgent(args=args)
    if args.load_ckpt:
        agent.load_checkpoint(args.load_ckpt)
    agent.run(episodes=args.episodes)