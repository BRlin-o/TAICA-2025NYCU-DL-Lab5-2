import os, json
import datetime
from copy import deepcopy

def save_config(agent, args, config_dict):
    """
    保存配置到 wandb_id 資料夾中，包含版本控制
    
    參數:
        agent: DQNAgent 實例
        args: 命令行參數
        config_dict: 發送到 wandb 的配置字典
    """
    if agent.save_dir is None:
        print("警告: 保存目錄未設置，無法保存配置")
        return
        
    # 將命令行參數轉換為字典
    args_dict = vars(args)
    
    # 創建完整配置字典
    full_config = {
        "args": args_dict,
        "wandb_config": config_dict,
        "runtime_info": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "wandb_id": agent.wandb_id,
            "is_resumed": agent.episode > 0,  # 如果 episode > 0，則為續訓
        }
    }
    
    # 保存最新配置
    config_path = os.path.join(agent.save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)
    print(f"最新配置已保存至 {config_path}")
    
    # 如果是首次訓練（非續訓），保存原始配置
    if agent.episode == 0:
        base_config_path = os.path.join(agent.save_dir, "base_config.json")
        with open(base_config_path, "w") as f:
            json.dump(full_config, f, indent=2)
        print(f"原始配置已保存至 {base_config_path}")
    
    # 如果是續訓且參數有變更，保存帶時間戳的配置（版本控制）
    elif agent.episode > 0:
        # 嘗試加載先前配置
        prev_config_path = os.path.join(agent.save_dir, "config.json")
        if os.path.exists(prev_config_path):
            try:
                with open(prev_config_path, "r") as f:
                    prev_config = json.load(f)
                
                # 檢查是否有重要參數變化（只檢查 args 部分）
                prev_args = prev_config.get("args", {})
                current_args = full_config["args"]
                
                # 定義重要參數列表（如果這些參數變化了，需要特別記錄）
                important_params = ["lr", "batch_size", "epsilon_decay", "target_update_frequency"]
                
                # 檢查重要參數是否有變化
                has_important_changes = False
                for param in important_params:
                    if param in prev_args and param in current_args:
                        if prev_args[param] != current_args[param]:
                            has_important_changes = True
                            break
                
                # 如果有重要變化，保存帶時間戳的配置
                if has_important_changes:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    versioned_config_path = os.path.join(agent.save_dir, f"config_{timestamp}.json")
                    with open(versioned_config_path, "w") as f:
                        json.dump(full_config, f, indent=2)
                    print(f"檢測到重要參數變化，新配置版本已保存至 {versioned_config_path}")
            except Exception as e:
                print(f"比較先前配置時出錯: {e}")
                
def load_config(agent):
    """從 wandb_id 資料夾加載最新配置"""
    if agent.save_dir is None:
        print("警告: 保存目錄未設置，無法加載配置")
        return None
        
    config_path = os.path.join(agent.save_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在")
        return None
        
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"配置已從 {config_path} 加載")
        return config
    except Exception as e:
        print(f"加載配置時出錯: {e}")
        return None