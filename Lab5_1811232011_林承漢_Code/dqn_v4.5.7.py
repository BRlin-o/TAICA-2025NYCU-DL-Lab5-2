# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os, random, time
import gymnasium as gym
try:
    from gymnasium.wrappers.frame_skip import FrameSkip
except ImportError:
    from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing as FrameSkip
import cv2
import math
import ale_py
from collections import deque
import wandb
import argparse
from typing import Deque, Tuple

# --- Additional imports for async DataLoader and vectorized env ---
import torch.backends.cudnn
from torch.utils.data import IterableDataset, DataLoader
from gymnasium.vector import AsyncVectorEnv

from utils import save_config, load_config

gym.register_envs(ale_py)


# --- Lightweight LazyFrames implementation for stacked Atari frames
class LazyFrames:
    """
    A reduced‑memory wrapper for stacking Atari frames.

    Instead of copying each 84×84 frame four times, we keep a list of
    references and only create the stacked np.ndarray when __array__()
    is called (e.g. right before feeding the network).
    """
    __slots__ = ("frames", "out")

    def __init__(self, frames):
        self.frames = frames        # list of np.uint8 H×W
        self.out = None             # cached np.ndarray

    def __array__(self, dtype=None, copy=None):
        if self.out is None:
            # Stack along channel dimension: (C, H, W)
            self.out = np.stack(self.frames, axis=0)
        result = self.out.astype(dtype) if dtype else self.out
        if copy:
            return result.copy()
        return result
    
    def __torch_tensor__(self):
        # 新增方法，實現 torch 轉換
        return torch.from_numpy(np.array(self))

    def __len__(self):
        return len(self.frames)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
    Deep Q-Network architecture with Dueling Network design
    
    Args:
        num_actions: Number of actions in the environment
        input_shape: Shape of input state (channels, height, width) or (features,)
        use_cnn: Whether to use convolutional layers (True for images)
    """
    def __init__(self, num_actions, input_shape, use_cnn=False):
        super(DQN, self).__init__()
        
        if use_cnn:
            # CNN feature extractor for image inputs
            self.features = nn.Sequential(
                # First conv layer: 32 filters, 8x8 kernel, stride 4
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                # Second conv layer: 64 filters, 4x4 kernel, stride 2
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                # Third conv layer: 64 filters, 3x3 kernel, stride 1
                nn.Conv2d(64, 128, kernel_size=3, stride=1),
                nn.ReLU(),
                # Flatten for fully connected layers
                nn.Flatten()
            )
            # Calculate output size after convolutions (64 channels, 7x7 spatial dimensions)
            feature_size = 128 * 7 * 7
        else:
            # MLP feature extractor for vector inputs
            self.features = nn.Sequential(
                nn.Linear(input_shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
            feature_size = 128
            
        # Dueling architecture
        # Value stream - estimates the value of the state
        self.value = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream - estimates the advantage of each action
        self.advantage = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        # Extract features from input
        features = self.features(x)
        
        # Compute state value
        value = self.value(features)
        
        # Compute advantages for each action
        advantage = self.advantage(features)
        
        # Combine value and advantage using dueling formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # This helps with stability by separating state value from action advantages
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        
    def preprocess(self, obs):
        # 裁剪無用區域 (通常Atari遊戲上方和下方有無用訊息)
        cropped = obs[34:194]
        # 轉灰度
        if cropped.ndim == 3 and cropped.shape[2] == 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        else:
            gray = cropped
        # 調整大小
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # 保持uint8類型以減少內存使用
        return resized.astype(np.uint8)
        
    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        # 使用LazyFrames優化內存
        return LazyFrames(list(self.frames))
        
    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        # 使用LazyFrames優化內存
        return LazyFrames(list(self.frames))



class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) implementation
    Stores transitions and samples them according to TD error
    See paper: https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha        # How much prioritization to use (α=0 is uniform sampling)
        self.beta = beta          # Importance sampling correction (β=1 is full correction)
        self.beta_start = beta    # Initial beta value
        self.buffer = []          # Store (s, a, r, s', done) tuples
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Priorities for each transition
        self.pos = 0              # Current position in buffer
        self.epsilon = 1e-6       # Small constant to avoid zero priority

    def add(self, transition, error=None):
        """Add a new transition to the buffer with priority"""
        # Set max priority for new transitions
        if error is None:
            priority = self.priorities.max() if len(self.buffer) else 1.0
        else:
            # Use TD error as priority with a small epsilon to ensure non-zero
            priority = abs(error) + self.epsilon

        # Add to buffer or update existing transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
            
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions based on priorities"""
        # Use only valid entries in buffer
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        # Calculate sampling probabilities
        scaled_prios = prios ** self.alpha
        probs = scaled_prios / scaled_prios.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        transitions = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert to torch Tensors for DataLoader pin_memory
        states = torch.from_numpy(np.array(states)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        actions = torch.from_numpy(np.array(actions)).long()
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32))
        dones = torch.from_numpy(np.array(dones, dtype=np.float32))
        weights = torch.tensor(weights, dtype=torch.float32)
        return states, actions, rewards, next_states, dones, indices, weights


# --- Iterable Dataset for DataLoader async sampling from PER ---
class ReplayDataset(IterableDataset):
    """Iterable Dataset that continuously samples from the PER buffer."""
    def __init__(self, memory, batch_size):
        super().__init__()
        self.memory = memory
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield self.memory.sample(self.batch_size)

    def update_priorities(self, indices, errors):
        """Update priorities of sampled transitions"""
        self.priorities[indices] = (np.abs(errors) + self.epsilon) ** self.alpha
    
    def __iter__(self):
        """Allow iterating over buffer contents"""
        return iter(self.buffer)
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)

        

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        # Create an asynchronous vectorized environment for parallel sampling
        num_envs = getattr(args, "num_envs", 8)
        env_fns = [lambda idx=i: FrameSkip(gym.make(env_name, render_mode="rgb_array", frameskip=1), frame_skip=args.frame_skip) for i in range(num_envs)]
        self.env = AsyncVectorEnv(env_fns)

        # self.test_env = gym.make(env_name, render_mode="rgb_array")
        test_base_env = gym.make(env_name, render_mode="rgb_array", frameskip=1)
        self.test_env = FrameSkip(test_base_env, frame_skip=args.frame_skip)
        
        self.num_actions = self.env.single_action_space.n

        # 確定是否為 Atari 環境（例如 Pong）
        self.is_atari = env_name.startswith("ALE/")

        # self.input_dim = self.env.observation_space.shape[0]
        # self.preprocessor = AtariPreprocessor()
        if self.is_atari:
            # Atari 環境使用圖像輸入
            self.preprocessor = AtariPreprocessor()
            self.input_shape = (4, 84, 84)  # 4 個堆疊的灰度圖像
        else:
            # CartPole 環境使用向量輸入
            self.preprocessor = AtariPreprocessor()  # 保持一致性，但實際上不會用於預處理
            self.input_shape = (self.env.single_observation_space.shape[0],)  # 例如 (4,) 對於 CartPole

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        print("Using device:", self.device)

        self.q_net = DQN(self.num_actions, self.input_shape, use_cnn=self.is_atari).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, self.input_shape, use_cnn=self.is_atari).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.RMSprop(self.q_net.parameters(), lr=args.lr, alpha=0.95, eps=0.01)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100000, gamma=0.5)

        # Prioritized replay buffer (Task‑3)
        self.memory = PrioritizedReplayBuffer(
            capacity=args.memory_size,
            alpha=args.per_alpha,
            beta=args.per_beta_start
        )
        # Wrap PER buffer with DataLoader for async data transfer
        self.batch_size = args.batch_size
        self.dataset = ReplayDataset(self.memory, self.batch_size)
        self.loader = DataLoader(self.dataset, batch_size=None, num_workers=4, pin_memory=True)
        self.loader_iter = iter(self.loader)

        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_start = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.linear_decay_steps = args.linear_decay_steps

        self.n_step = args.n_step
        self.gamma_n = self.gamma ** self.n_step
        self.nstep_queue: Deque[Tuple] = deque(maxlen=self.n_step)

        self.episode = 0
        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0 if not self.is_atari else -21  # Initilized to 0 for CartPole and to -21 for Pong

        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.base_save_dir = args.save_dir
        os.makedirs(self.base_save_dir, exist_ok=True)

        self.wandb_id = None
        self.save_dir = None

    def set_save_dir(self, wandb_id):
        """設置基於 wandb_id 的保存目錄"""
        self.wandb_id = wandb_id
        self.save_dir = os.path.join(self.base_save_dir, wandb_id)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"保存目錄設置為: {self.save_dir}")
        return self.save_dir

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        # state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if self.is_atari:
            state_tensor = state_tensor / 255.0
        state_tensor = state_tensor.to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()
    
    def collect_experience(self, state, action, reward, next_state, done):
        """
        Collect experience and add to replay buffer with n-step returns
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add current transition to n-step queue
        self.nstep_queue.append((state, action, reward))
        
        # When queue is full, compute n-step return and add to replay buffer
        if len(self.nstep_queue) == self.n_step:
            # Compute n-step return: R_n = r_0 + gamma*r_1 + ... + gamma^(n-1)*r_(n-1)
            R_n = sum((self.gamma ** i) * self.nstep_queue[i][2] 
                    for i in range(self.n_step))
            
            # Get initial state and action
            state_0, action_0, _ = self.nstep_queue[0]
            
            # Add to replay buffer
            self.memory.add((state_0, action_0, R_n, next_state, done))
            
            # Remove oldest transition (we keep the queue at n steps)
            self.nstep_queue.popleft()
        
        # Handle episode termination - process remaining transitions in queue
        if done:
            # Process all remaining transitions in queue
            while self.nstep_queue:
                # Compute return from remaining steps
                R_n = sum((self.gamma ** i) * self.nstep_queue[i][2] 
                        for i in range(len(self.nstep_queue)))
                
                # Get state and action from the oldest remaining transition
                state_0, action_0, _ = self.nstep_queue[0]
                
                # Add to buffer
                self.memory.add((state_0, action_0, R_n, next_state, done))
                
                # Remove processed transition
                self.nstep_queue.popleft()

    def run(self, episodes=1000, checkpoint_path=None, checkpoint_interval=100000):
        self.load_checkpoint(checkpoint_path)
        while self.episode < episodes:
            obs, _ = self.env.reset()

            if self.is_atari:
                state = self.preprocessor.reset(obs)
            else:
                state = obs  # CartPole 直接使用原始狀態
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if self.is_atari:
                    next_state = self.preprocessor.step(next_obs)
                else:
                    next_state = next_obs
                
                # self.memory.append((state, action, reward, next_state, done))
                # --- n‑step buffering & PER insert ---
                self.nstep_queue.append((state, action, reward))
                if len(self.nstep_queue) == self.n_step:
                    R_n = sum((self.gamma ** i) * self.nstep_queue[i][2] for i in range(self.n_step))
                    state_0, action_0, _ = self.nstep_queue[0]
                    # 計算TD誤差
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(np.array(state_0)).float().unsqueeze(0)
                        if self.is_atari:
                            state_tensor = state_tensor / 255.0
                        state_tensor = state_tensor.to(self.device)
                        q_value = self.q_net(state_tensor)[0, action_0].item()
                        error = R_n - q_value
                    
                    self.memory.add((state_0, action_0, R_n, next_state, done), error)
                    # self.memory.add((state_0, action_0, R_n, next_state, done))
                if done:
                    while self.nstep_queue:
                        R_n = sum((self.gamma ** i) * self.nstep_queue[i][2] for i in range(len(self.nstep_queue)))
                        state_0, action_0, _ = self.nstep_queue[0]
                        self.memory.add((state_0, action_0, R_n, next_state, done))
                        self.nstep_queue.popleft()

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0 and self.env_count > 0: # Log every 1000 steps to wandb
                    print(f"[Collect] Ep: {self.episode} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    ########## YOUR CODE HERE  ##########
                    wandb.log({
                        "env_step": self.env_count, # 當前環境交互的總步數
                        "progress/episode": self.episode, # 當前訓練回合數
                        "progress/step_count": step_count, # 當前回合已執行的步數
                        "progress/env_step_count": self.env_count, # 與環境交互的總步數（累積值）
                        "progress/update_count": self.train_count, # 網絡更新（梯度下降）總次數
                        "agent/epsilon": self.epsilon,
                        "agent/buffer_size": len(self.memory) # 經驗回放緩衝區的樣本數量
                    })
                    ########## END OF YOUR CODE ##########   

                if self.env_count % 10000 == 0 and self.env_count > 0:
                    eval_reward, eval_std = self.evaluate()

                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        self.save_checkpoint("best_model.pt", is_best=True)
                        print(f"Saved new best model with reward {eval_reward}")

                    print(f"[TrueEval] Ep: {self.episode} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                    wandb.log({
                        "env_step": self.env_count,
                        "progress/env_step_count": self.env_count,
                        "progress/update_count": self.train_count,
                        "evaluation/episode_reward": eval_reward,  # 關鍵指標，區分於訓練獎勵
                        "evaluation/best_reward": self.best_reward
                    })

                    # task requirement(T1)
                    wandb.log({
                        "env_step": self.env_count,
                        "Episode Reward vs Env Steps": eval_reward  # 直接用這個名稱便於在Wandb中找到
                    })

                if self.env_count % checkpoint_interval == 0:
                    self.save_checkpoint(f"model_step{self.env_count}.pt", is_periodic=True)
            
            print(f"[Eval] Ep: {self.episode} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            ########## YOUR CODE HERE  ##########
            wandb.log({
                "env_step": self.env_count,
                "progress/episode": self.episode,
                "performance/episode_reward": total_reward,  # 關鍵指標，用於繪製圖表
                "progress/env_step_count": self.env_count,
                "progress/update_count": self.train_count,
                "agent/epsilon": self.epsilon,
                "performance/episode_length": step_count
            })
            ########## END OF YOUR CODE ##########  

            

            # Save checkpoints with adaptive frequency based on episode number
            if (self.episode < 250 and self.episode % 50 == 0 and self.episode > 0) or \
                (250 <= self.episode < 500 and self.episode % 25 == 0) or \
                (500 <= self.episode < 1000 and self.episode % 10 == 0) or \
                (self.episode >= 1000 and self.episode % 5 == 0):
                 self.save_checkpoint("latest.pt")  # Save latest checkpoint

            self.episode += 1

    def evaluate(self, num_episodes=5):
        """進行多次評估並返回平均分數"""
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.test_env.reset()
            if self.is_atari:
                state = self.preprocessor.reset(obs)
            else:
                state = obs
            done = False
            episode_reward = 0
            
            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
                # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                if self.is_atari:
                    state_tensor = state_tensor / 255.0
                state_tensor = state_tensor.to(self.device)
                
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                    
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if self.is_atari:
                    state = self.preprocessor.step(next_obs)
                else:
                    state = next_obs
                    
            total_rewards.append(episode_reward)
            
        avg_reward = sum(total_rewards) / len(total_rewards)
        std_reward = np.std(total_rewards)
        
        return avg_reward, std_reward


    def train(self):
        """Train the DQN on a batch of experiences from replay buffer"""
        # Skip training if we don't have enough samples yet
        if len(self.memory) < self.replay_start_size:
            return 
        
        # Update exploration epsilon with linear decay
        if self.epsilon > self.epsilon_min:
            decay_rate = (self.epsilon_start - self.epsilon_min) / self.linear_decay_steps
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_rate)
        
        self.train_count += 1
        
        # Load batch from DataLoader for async CPU->GPU transfer
        states, actions, rewards, next_states, dones, indices, weights = next(self.loader_iter)
        states = states.to(self.device, non_blocking=True)
        next_states = next_states.to(self.device, non_blocking=True)
        if self.is_atari:
            states = states / 255.0
            next_states = next_states / 255.0
        actions = actions.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True)
        dones = dones.to(self.device, non_blocking=True)
        weights = weights.to(self.device, non_blocking=True)
        
        # Get current Q values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values using Double DQN approach
        with torch.no_grad():  # Don't need gradients for target computation
            # Action selection using online network
            next_actions = self.q_net(next_states).argmax(1)
            # Value estimation using target network
            next_q_target = self.target_net(next_states) \
                                .gather(1, next_actions.unsqueeze(1)) \
                                .squeeze(1)
            # Compute target using n-step return
            target_q_values = rewards + (1 - dones) * self.gamma_n * next_q_target
        
        # Compute TD errors for updating priorities
        td_errors = target_q_values - q_values
        
        # Compute weighted Huber loss
        loss_elements = F.smooth_l1_loss(q_values, target_q_values, reduction='none')
        loss = (weights * loss_elements).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1)
        self.optimizer.step()
        
        # Update sample priorities in replay buffer
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        self.memory.beta = min(1.0, self.memory.beta_start +
                (1.0 - self.memory.beta_start) * self.env_count / self.linear_decay_steps)
        
        # Periodically update the target network
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Periodically log training statistics
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f}, Q mean: {q_values.mean().item():.3f}, std: {q_values.std().item():.3f}")
            wandb.log({
                "train_step": self.train_count,
                "train/loss": loss.item(),
                "train/q_mean": q_values.mean().item(),
                "train/q_std": q_values.std().item(),
                "train/beta": self.memory.beta
            })
        
        # Update learning rate if scheduler is defined
        if self.train_count % 10 == 0:  # Update LR less frequently
            self.lr_scheduler.step()

    def save_checkpoint(self, name="latest.pt", is_best=False, is_periodic=False):
        """
        保存檢查點到當前 wandb_id 的目錄
        
        參數:
            name: 檔案名稱
            is_best: 是否是最佳模型
            is_periodic: 是否是定期保存
        """
        if self.save_dir is None:
            print("警告: 保存目錄未設置，無法保存檢查點")
            return
            
        path = os.path.join(self.save_dir, name)
        
        ckpt = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optim": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "env_count": self.env_count,
            "train_count": self.train_count,
            "best_reward": self.best_reward,
            # "memory": list(self.memory),
            "episode": self.episode,
            "wandb_id": self.wandb_id,
            "is_atari": self.is_atari,
        }
        torch.save(ckpt, path)
        
        # 記錄保存類型
        save_type = "最佳模型" if is_best else ("定期保存" if is_periodic else "最新狀態")
        print(f"{save_type}檢查點已保存至 {path}")
        
        # 同時更新 latest.pt（如果當前保存的不是 latest.pt）
        if name != "latest.pt":
            latest_path = os.path.join(self.save_dir, "latest.pt")
            torch.save(ckpt, latest_path)

    def find_latest_checkpoint(self):
        """尋找當前 wandb_id 目錄下的最新檢查點"""
        if self.save_dir is None or not os.path.exists(self.save_dir):
            return None
            
        latest_path = os.path.join(self.save_dir, "latest.pt")
        if os.path.exists(latest_path):
            return latest_path
            
        # 如果 latest.pt 不存在，尋找最新的週期性保存
        checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith(".pt")]
        if not checkpoints:
            return None
            
        # 找出時間戳最新的檢查點
        latest_checkpoint = max(checkpoints, key=lambda f: os.path.getmtime(os.path.join(self.save_dir, f)))
        return os.path.join(self.save_dir, latest_checkpoint)

    def load_checkpoint(self, path=None):
        """
        從檢查點文件加載訓練狀態
        
        參數:
            path: 檢查點文件路徑，如果為 None，嘗試加載最新的檢查點
        """
        if path is None:
            path = self.find_latest_checkpoint()
            if path is None:
                print("未找到可加載的檢查點")
                return False
                
        if not os.path.exists(path):
            print(f"檢查點文件 {path} 不存在")
            return False
            
        try:
            ckpt = torch.load(path, map_location=self.device)
            
            # 加載模型參數
            self.q_net.load_state_dict(ckpt["q_net"])
            self.target_net.load_state_dict(ckpt["target_net"])
            self.optimizer.load_state_dict(ckpt["optim"])
            
            # 恢復訓練狀態
            self.epsilon = ckpt["epsilon"]
            self.env_count = ckpt["env_count"]
            self.train_count = ckpt["train_count"]
            self.best_reward = ckpt.get("best_reward", 0)
            self.episode = ckpt.get("episode", 0)
            
            # 檢查環境類型是否匹配
            ckpt_is_atari = ckpt.get("is_atari", None)
            if ckpt_is_atari is not None and ckpt_is_atari != self.is_atari:
                print(f"警告: 檢查點環境類型 ({ckpt_is_atari}) 與當前環境類型 ({self.is_atari}) 不匹配!")
            
            # 恢復經驗回放緩衝區
            if "memory" in ckpt:
                self.memory = deque(ckpt["memory"], maxlen=self.memory.maxlen)
            
            # 獲取 wandb_id 並設置保存目錄
            wandb_id = ckpt.get("wandb_id", None)
            if wandb_id and wandb_id != self.wandb_id:
                self.set_save_dir(wandb_id)
            
            print(f"檢查點已從 {path} 加載")
            print(f"恢復至: 回合={self.episode}, 環境步數={self.env_count}, 訓練次數={self.train_count}")
            return True
        except Exception as e:
            print(f"加載檢查點時出錯: {e}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", 
                            help="環境名稱") # ["CartPole-v1", "ALE/Pong-v5"]
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="pong-run")
    parser.add_argument("--wandb-project", type=str, default="DLP-Lab5-DQN-Pong(T2) v2")
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=0.4)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.025)
    parser.add_argument("--episodes", type=int, default=4000, help="訓練回合數")
    parser.add_argument("--target-update-frequency", type=int, default=500)
    parser.add_argument("--replay-start-size", type=int, default=20000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=50000)
    parser.add_argument("--frame-skip", type=int, default=4, help="Atari環境跳過的幀數")
    parser.add_argument("--linear-decay-steps", type=int, default=1000000,
                            help="ε 由 ε_start 線性降到 ε_min 所需的 env steps 數")
    parser.add_argument("--n-step", type=int, default=5,
                        help="number of steps for n‑step return (Task‑3)")
    parser.add_argument("--per-alpha", type=float, default=0.6,
                        help="PER exponent α")
    parser.add_argument("--per-beta-start", type=float, default=0.4,
                        help="initial β for PER importance‑sampling weights")
    args = parser.parse_args()

    def check_env(env_name):
        return "CartPole".lower() in env_name.lower()

    config = {
        "algorithm": "DQN",
        "environment": args.env_name,
        "batch_size": args.batch_size,
        "memory_size": args.memory_size,
        "learning_rate": args.lr,
        "discount_factor": args.discount_factor,
        "epsilon_start": args.epsilon_start,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "target_update_frequency": args.target_update_frequency,
        "replay_start_size": args.replay_start_size,
        "max_episode_steps": args.max_episode_steps,
        "train_per_step": args.train_per_step,
        "frame_skip": args.frame_skip,
        "linear_decay_steps": args.linear_decay_steps,
        "n_step": args.n_step,
        "per_alpha": args.per_alpha,
        "per_beta_start": args.per_beta_start,
        "epsilon_schedule": "linear",
        "architecture": "2-layer MLP (128, 128)" if check_env(args.env_name) else "CNN",
        "optimizer": "Adam",
        "loss_function": "MSE"
    }

    agent = DQNAgent(env_name=args.env_name, args=args)

    # args.wandb_id = wandb.run.id
    if args.wandb_id:
        agent.set_save_dir(args.wandb_id)
        agent.load_checkpoint()  # 嘗試加載最新檢查點
        loaded_config = load_config(agent)  # 加載配置
        
        # 如果加載了配置，檢查關鍵參數是否變更
        if loaded_config and "args" in loaded_config:
            loaded_args = loaded_config["args"]
            print("已加載之前的配置。以下是主要參數:")
            for key in ["lr", "batch_size", "epsilon_decay", "target_update_frequency"]:
                if key in loaded_args:
                    current_value = getattr(args, key)
                    loaded_value = loaded_args[key]
                    if current_value != loaded_value:
                        print(f"警告: 參數 '{key}' 已從 {loaded_value} 變更為 {current_value}")
                    else:
                        print(f"參數 '{key}' 保持不變: {current_value}")

    wandb.init(
        id=args.wandb_id,                          # 使用之前的運行 ID
        resume="allow" if args.wandb_id else None, # 允許續接
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config,
        save_code=True,
        tags=["task1", "cartpole", "dqn"] if check_env(args.env_name) else ["task2", "pong", "dqn"],
    )

    wandb.define_metric("env_step")
    wandb.define_metric("train_step")
    wandb.define_metric("progress/*",   step_metric="env_step")
    wandb.define_metric("performance/*", step_metric="env_step")
    wandb.define_metric("evaluation/*",  step_metric="env_step")
    wandb.define_metric("train/*",      step_metric="train_step")
    
    agent.set_save_dir(wandb.run.id)

    wandb.watch(agent.q_net, log="gradients", log_freq=1000)

    save_config(agent, args, config)

    try:
        agent.run(episodes=args.episodes, checkpoint_interval=args.checkpoint_interval)
    except KeyboardInterrupt:
        print("Training interrupted. Saving latest checkpoint...")
        agent.save_checkpoint("latest.pt")
    finally:
        # 在訓練結束時保存最終檢查點
        agent.save_checkpoint("latest.pt")
        wandb.finish()