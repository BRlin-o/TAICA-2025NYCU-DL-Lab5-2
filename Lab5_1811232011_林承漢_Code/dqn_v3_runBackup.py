# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, random, time
import gymnasium as gym
import cv2
import ale_py
from collections import deque
import wandb
import argparse
from typing import Deque, Tuple

from utils import save_config, load_config

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
    def __init__(self, num_actions, input_shape, use_cnn=False):
        super(DQN, self).__init__()
        ## Lab1
        if not use_cnn:            # CartPole → 向量
            self.network = nn.Sequential(
                nn.Linear(input_shape[0], 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, num_actions)
            )
        ## Lab2, 3
        else:                                # Pong → 影像 (4,84,84)
            self.network = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),     nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),     nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64*7*7, 512),      nn.ReLU(),
                nn.Linear(512, num_actions)
            )

    def forward(self, x):
        return self.network(x)


# class AtariWrapper:
#     """包裝Atari環境，實現幀跳過和終止管理"""
#     def __init__(self, env, skip=4):
#         self.env = env
#         self.skip = skip
#         self.lives = 0
        
#     def reset(self):
#         obs, info = self.env.reset()
#         self.lives = info.get('lives', 0)
#         return obs, info
        
#     def step(self, action):
#         total_reward = 0.0
#         done = False
#         info = {}
        
#         # 跳過frames，但保持同一個動作
#         for i in range(self.skip):
#             obs, reward, terminated, truncated, info = self.env.step(action)
#             total_reward += reward
#             done = terminated or truncated
            
#             # Pong特有：檢測生命損失作為"done"信號
#             current_lives = info.get('lives', 0)
#             if current_lives < self.lives:
#                 done = True
#                 self.lives = current_lives
                
#             if done:
#                 break
                
#         return obs, total_reward, terminated, truncated, info
class AtariWrapper:
    """包装Atari环境，实现帧跳过和终止管理"""
    def __init__(self, env, skip=4):
        self.env = env
        self.skip = skip
        
    def reset(self):
        return self.env.reset()
        
    def step(self, action):
        total_reward = 0.0
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        cropped = obs[35:195]
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        return normalized

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

        # 確定是否為 Atari 環境（例如 Pong）
        self.is_atari = env_name.startswith("ALE/")

        # self.input_dim = self.env.observation_space.shape[0]
        # self.preprocessor = AtariPreprocessor()
        if self.is_atari:
            # Atari 環境使用圖像輸入
            self.env = AtariWrapper(self.env, skip=args.frame_skip)
            self.test_env = AtariWrapper(self.test_env, skip=args.frame_skip)
            self.preprocessor = AtariPreprocessor()
            self.input_shape = (4, 84, 84)  # 4 個堆疊的灰度圖像
        else:
            # CartPole 環境使用向量輸入
            self.preprocessor = AtariPreprocessor()  # 保持一致性，但實際上不會用於預處理
            self.input_shape = (self.env.observation_space.shape[0],)  # 例如 (4,) 對於 CartPole

        

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print("Using device:", self.device)


        self.q_net = DQN(self.num_actions, self.input_shape, use_cnn=self.is_atari).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, self.input_shape, use_cnn=self.is_atari).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # 創建經驗回放緩衝區 (使用列表而不是單獨的類)
        self.memory = deque(maxlen=args.memory_size)


        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_start = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

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
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000, checkpoint_path=None, checkpoint_interval=10000):
        self.load_checkpoint(checkpoint_path)
        while self.episode < episodes:
            obs, _ = self.env.reset()

            if self.is_atari:
                state = self.preprocessor.reset(obs)
            else:
                state = obs  # CartPole 直接使用原始狀態
            done = False
            total_reward = 0
            step_count = 0 ## 環境步數

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if self.is_atari:
                    next_state = self.preprocessor.step(next_obs)
                else:
                    next_state = next_obs
                
                self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    # print(f"Calling train() - Memory: {len(self.memory)}/{self.replay_start_size}")
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0: # Log every 1000 steps to wandb
                    print(f"[Collect] Ep: {self.episode} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    ########## YOUR CODE HERE  ##########
                    wandb.log({
                        "progress/episode": self.episode, # 當前訓練回合數
                        "progress/step_count": step_count, # 當前回合已執行的步數
                        "progress/env_step_count": self.env_count, # 與環境交互的總步數（累積值）
                        "progress/update_count": self.train_count, # 網絡更新（梯度下降）總次數
                        "agent/epsilon": self.epsilon,
                        "agent/buffer_size": len(self.memory) # 經驗回放緩衝區的樣本數量
                    }, step=self.env_count)
                    ########## END OF YOUR CODE ##########   
            
            print(f"[Eval] Ep: {self.episode} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            ########## YOUR CODE HERE  ##########
            wandb.log({
                "progress/episode": self.episode,
                "performance/episode_reward": total_reward,  # 關鍵指標，用於繪製圖表
                "progress/env_step_count": self.env_count,
                "progress/update_count": self.train_count,
                "agent/epsilon": self.epsilon,
                "performance/episode_length": step_count
            }, step=self.env_count)
            ########## END OF YOUR CODE ##########  

            if self.env_count % checkpoint_interval == 0:
                self.save_checkpoint(f"model_step{self.env_count}.pt", is_periodic=True)

            if self.episode % 20 == 0:
                eval_reward = self.evaluate()

                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.save_checkpoint("best_model.pt", is_best=True)
                    print(f"Saved new best model with reward {eval_reward}")

                print(f"[TrueEval] Ep: {self.episode} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "progress/env_step_count": self.env_count,
                    "progress/update_count": self.train_count,
                    "evaluation/episode_reward": eval_reward,  # 關鍵指標，區分於訓練獎勵
                    "evaluation/best_reward": self.best_reward
                }, step=self.env_count)

                # task requirement(T1)
                wandb.log({
                    "Episode Reward vs Env Steps": eval_reward  # 直接用這個名稱便於在Wandb中找到
                }, step=self.env_count)

            # if self.episode % 100 == 0:
            #     self.save_checkpoint("latest.pt")  # Save latest checkpoint

            self.episode += 1

    def evaluate(self):
        obs, _ = self.test_env.reset()
        # 根據環境類型預處理狀態
        if self.is_atari:
            state = self.preprocessor.reset(obs)  # Atari環境使用預處理器
        else:
            state = obs  # CartPole直接使用原始狀態向量
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            # 根據環境類型處理下一個狀態
            if self.is_atari:
                state = self.preprocessor.step(next_obs)  # Atari環境使用預處理器
            else:
                state = next_obs  # CartPole直接使用原始狀態向量

        return total_reward


    def train(self):
        # print(f"Memory size: {len(self.memory)}/{self.replay_start_size}")
        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min, 
                self.epsilon - (self.epsilon_start - self.epsilon_min) / args.linear_decay_steps
            )
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # 計算下一個狀態的最大Q值（不需要梯度）
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            # 計算目標Q值
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # 計算損失（Bellman誤差）
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # 梯度下降優化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")

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
            "memory": list(self.memory),
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
            print(f"正在加載最新檢查點: {path}")
            if path is None:
                print("未找到可加載的檢查點")
                return False
                
        if not os.path.exists(path):
            print(f"檢查點文件 {path} 不存在")
            return False
            
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="環境名稱") # ["CartPole-v1", "ALE/Pong-v5"]
    parser.add_argument("--save-dir", type=str, default="./results/task2")
    parser.add_argument("--wandb-run-name", type=str, default="pong-run")
    parser.add_argument("--wandb-project", type=str, default="DLP-Lab5-DQN-Pong(T1)")
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--episodes", type=int, default=4000, help="訓練回合數")
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=5000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--frame-skip", type=int, default=4, help="Atari環境跳過的幀數")
    parser.add_argument("--linear-decay-steps", type=int, default=1000000, help="線性衰減的總步數")
    parser.add_argument("--use-dueling", action="store_true", help="是否使用Dueling架構")
    
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