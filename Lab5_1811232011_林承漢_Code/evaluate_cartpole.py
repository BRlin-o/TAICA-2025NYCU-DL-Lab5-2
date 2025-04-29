import numpy as np
import gymnasium as gym
import argparse
import os
from collections import deque
import cv2
import imageio
import sys
import time

# 從您的訓練腳本導入 DQN 類
from dqn_v2 import DQN, AtariPreprocessor

def evaluate_model(model_path, output_dir="./eval_results", episodes=5, render=True):
    """
    評估已訓練模型的性能並可選擇性地記錄視頻
    
    參數:
        model_path: 訓練好的模型檢查點路徑
        output_dir: 輸出目錄
        episodes: 評估的回合數
        render: 是否渲染並保存視頻
    """
    # 確定使用的設備
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"使用設備: {device}")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建環境
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    num_actions = env.action_space.n
    input_shape = (env.observation_space.shape[0],)  # CartPole的輸入形狀
    
    # 載入模型 - 使用多種方法嘗試載入
    try:
        # 嘗試方法1: weights_only=False
        print("嘗試載入方法1: weights_only=False")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        print("載入成功！")
    except Exception as e:
        print(f"載入方法1失敗: {e}")
        try:
            # 嘗試方法2: weights_only=True
            print("嘗試載入方法2: weights_only=True")
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
            print("載入成功！")
        except Exception as e2:
            print(f"載入方法2失敗: {e2}")
            try:
                # 嘗試方法3: 使用safe_globals
                print("嘗試載入方法3: 使用safe_globals")
                import torch.serialization
                with torch.serialization.safe_globals(['numpy.core.multiarray._reconstruct']):
                    ckpt = torch.load(model_path, map_location=device)
                print("載入成功！")
            except Exception as e3:
                # 嘗試方法4: 不使用任何選項
                print(f"載入方法3失敗: {e3}")
                print("嘗試載入方法4: 不使用額外選項")
                ckpt = torch.load(model_path, map_location=device)
                print("載入成功！")
    
    print(f"檢查點類型: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"檢查點鍵: {list(ckpt.keys())}")
    
    # 根據檢查點類型決定如何載入模型
    if isinstance(ckpt, dict) and "q_net" in ckpt:
        # 完整檢查點格式
        print("從完整檢查點載入模型權重")
        model = DQN(num_actions, input_shape, use_cnn=False).to(device)
        model.load_state_dict(ckpt["q_net"])
    else:
        # 嘗試直接使用檢查點或其它格式
        print("從權重字典載入模型")
        model = DQN(num_actions, input_shape, use_cnn=False).to(device)
        if isinstance(ckpt, dict) and any(k.endswith(".weight") for k in ckpt.keys()):
            # 如果檢查點看起來像是狀態字典
            model.load_state_dict(ckpt)
        else:
            # 如果檢查點本身就是模型
            model = ckpt
    
    model.eval()
    print("模型已設置為評估模式")
    
    # 評估性能
    all_rewards = []
    all_episode_lengths = []
    
    print(f"開始評估 {episodes} 回合...")
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        frames = []
        
        while not done:
            # 保存當前幀
            if render:
                frame = env.render()
                frames.append(frame)
            
            # 選擇動作
            state_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            # 執行動作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新狀態和獎勵
            obs = next_obs
            total_reward += reward
            steps += 1
        
        all_rewards.append(total_reward)
        all_episode_lengths.append(steps)
        
        # 保存視頻
        if render and frames:
            video_path = os.path.join(output_dir, f"cartpole_eval_ep{ep}.mp4")
            with imageio.get_writer(video_path, fps=30) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"回合 {ep}: 獎勵 = {total_reward}, 步數 = {steps}, 視頻保存至 {video_path}")
        else:
            print(f"回合 {ep}: 獎勵 = {total_reward}, 步數 = {steps}")
    
    # 計算和打印統計數據
    avg_reward = np.mean(all_rewards)
    avg_length = np.mean(all_episode_lengths)
    print(f"\n評估完成! 共 {episodes} 回合")
    print(f"平均獎勵: {avg_reward:.2f}")
    print(f"平均回合長度: {avg_length:.2f}")
    print(f"獎勵標準差: {np.std(all_rewards):.2f}")
    
    # 將結果寫入文件
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write(f"模型: {model_path}\n")
        f.write(f"回合數: {episodes}\n")
        f.write(f"平均獎勵: {avg_reward:.2f}\n")
        f.write(f"平均回合長度: {avg_length:.2f}\n")
        f.write(f"獎勵標準差: {np.std(all_rewards):.2f}\n")
        f.write("\n詳細回合數據:\n")
        for ep in range(episodes):
            f.write(f"回合 {ep}: 獎勵 = {all_rewards[ep]}, 步數 = {all_episode_lengths[ep]}\n")
    
    print(f"評估結果已保存至 {results_file}")
    
    return avg_reward, avg_length, all_rewards, all_episode_lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="訓練好的模型檢查點路徑")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="輸出目錄")
    parser.add_argument("--episodes", type=int, default=5, help="評估的回合數")
    parser.add_argument("--no-render", action="store_true", help="不渲染視頻")
    args = parser.parse_args()
    
    print("開始評估模型...")
    print(f"模型路徑: {args.model_path}")
    print(f"輸出目錄: {args.output_dir}")
    print(f"評估回合數: {args.episodes}")
    print(f"渲染視頻: {not args.no_render}")
    
    try:
        evaluate_model(
            model_path=args.model_path,
            output_dir=args.output_dir,
            episodes=args.episodes,
            render=not args.no_render
        )
    except Exception as e:
        print(f"評估過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()