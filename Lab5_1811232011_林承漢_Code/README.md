# Lab5 1811232011 林承漢

## Installation

- MacOS / Linux:

```bash
python -m venv .venv
source ./.venv/bin/activate

## Normal
pip install -r requirements.txt
## Using CUDA
pip install -r requirements_CUDA.txt
```

- Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate.bat

## Normal
pip install -r requirements.txt
## Using CUDA
pip install -r requirements_CUDA.txt
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```


## Run

### Task1

```bash
python dqn_v2.py --env-name="CartPole-v1" \
    --save-dir="./results" \
    --wandb-project="DLP-Lab5-Task1" \
    --wandb-run-name="cartpole-basic" \
    --batch-size=64 \
    --memory-size=100000 \
    --lr=0.0005 \
    --episodes=10000
```

```bash
python dqn_v2.py --env-name="CartPole-v1" \
  --wandb-project="DLP-Lab5-Task1" \
  --wandb-run-name="cartpole-p2" \
  --batch-size=128 \
  --memory-size=50000 \
  --lr=0.0001 \
  --epsilon-decay=0.998 \
  --target-update-frequency=50 \
  --replay-start-size=1000 \
  --max-episode-steps=500 \
  --episodes=1000
```

### Task2

- v3

```bash
python dqn_v3.py --env-name "ALE/Pong-v5" \
    --wandb-project "DLP-Lab5-DQN-Pong(T2)-MPX" \
    --wandb-run-name "pong-optimized-4" \
    --batch-size 32 \
    --memory-size 200000 \
    --lr 0.00025 \
    --linear-decay-steps 1000000 \
    --frame-skip 4 \
    --train-per-step 2 \
    --target-update-frequency 10000 \
    --replay-start-size 5000 \
    --episodes 4000
```

- v4

```bash
python dqn_v4.py --env-name "ALE/Pong-v5" \
    --wandb-project "DLP-Lab5-DQN-Pong(T2v4)-MPS" \
    --wandb-run-name "pong-fskip" \
    --batch-size 32 \
    --memory-size 200000 \
    --lr 0.00025 \
    --linear-decay-steps 500000 \
    --frame-skip 4 \
    --train-per-step 2 \
    --target-update-frequency 10000 \
    --replay-start-size 25000 \
    --episodes 4000
```

- v4.2

```bash
python dqn_v4.2.py \
  --env-name "ALE/Pong-v5" \
  --wandb-project "DLP-Lab5-DQN-Pong(T3)-MPS" \
  --wandb-run-name "pong-enhanced-v4.2" \
  --batch-size 64 \
  --memory-size 200000 \
  --lr 2.25e-4 \
  --linear-decay-steps 500000 \
  --frame-skip 4 \
  --train-per-step 4 \
  --target-update-frequency 2000 \
  --replay-start-size 50000 \
  --n-step 5 \
  --per-alpha 0.7 \
  --per-beta-start 0.5 \
  --episodes 5000
```
- --batch-size 32             32 對 84×84×4 圖像在 M4 Pro 48 GB 最穩
- --memory-size 200000        20 萬筆 ≈ 13 GB UMA，可全放 RAM
- --lr 6.25e-5                2.5e-4 / 4 —— 文獻建議 PER + n-step 時降 LR
- --linear-decay-steps 100    ε、β 線性退火到 1 M env-steps（與論文同步）
- --frame-skip 4              Atari 標準
- --train-per-step 1          每與環境互動 1 步就更新 1 次（PER 已提高 sample-efficiency）
- --target-update-frequenc    Double DQN：較頻繁同步 target 可減 bias
- --replay-start-size 5000    先 warm-up 5 萬步再訓練（確保樣本多樣）
- --n-step 3                  ★ Task-3: n-step return
- --per-alpha 0.6             ★ Task-3: PER exponent α
- --per-beta-start 0.4        ★ Task-3: 初始 β，之後程式會線性漸進到 1
- --episodes 5000             保持 4 K–5 K 回合就能到 1 M steps

- v4.3

```bash
python dqn_v4.3.py \
  --env-name "ALE/Pong-v5" \
  --wandb-project "DLP-Lab5-DQN-Pong(T3)" \
  --wandb-run-name "pong-enhanced(duel+noisy+drq)-2" \
  --batch-size 32 \
  --memory-size 200000 \
  --lr 6.25e-5 \
  --frame-skip 4 \
  --train-per-step 4 \
  --target-update-frequency 5000 \
  --replay-start-size 20000 \
  --linear-decay-steps 1000000 \
  --n-step 5 \
  --per-alpha 0.5 \
  --per-beta-start 0.4 \
  --episodes 5000
```

- v4.4

```bash
python dqn_v4.4.py \
  --env-name "ALE/Pong-v5" \
  --wandb-project "DLP-Lab5-DQN-Pong(T3)-MPS" \
  --wandb-run-name "pong-enhanced-v4.4" \
  --batch-size 64 \
  --memory-size 200000 \
  --lr 2.25e-4 \
  --linear-decay-steps 500000 \
  --frame-skip 4 \
  --train-per-step 4 \
  --target-update-frequency 2000 \
  --replay-start-size 50000 \
  --n-step 5 \
  --per-alpha 0.7 \
  --per-beta-start 0.5 \
  --episodes 5000
```

- v4.4(m4pro)

```bash
python dqn_v4.4.py \
  --env-name="ALE/Pong-v5" \
  --batch-size=64 \
  --memory-size=300000 \
  --lr=0.00025 \
  --epsilon-decay=0.99999 \
  --target-update-frequency=2000 \
  --replay-start-size=20000 \
  --max-episode-steps=5000 \
  --train-per-step=1 \
  --frame-skip=4 \
  --linear-decay-steps=500000 \
  --n-step=3 \
  --per-alpha=0.6 \
  --per-beta-start=0.4 \
  --episodes=1000
```

- v4.4(RTX3090)

```bash
python dqn_v4.4.py
  --env-name="ALE/Pong-v5"
  --batch-size=128
  --memory-size=500000
  --lr=0.00025
  --epsilon-decay=0.99999
  --target-update-frequency=2000
  --replay-start-size=30000
  --max-episode-steps=10000
  --train-per-step=2
  --frame-skip=4
  --linear-decay-steps=800000
  --n-step=4
  --per-alpha=0.6
  --per-beta-start=0.4
  --episodes=2000
```

- v4.4(RTX4080)

```bash
python dqn_v4.4.py 
  --env-name "ALE/Pong-v5"
  --wandb-project "DLP-Lab5-DQN-Pong(T3)-4080"
  --wandb-run-name "pong-enhanced-v4.4"
  --batch-size 128
  --memory-size 500000
  --lr 0.00025
  --epsilon-decay 0.99999
  --target-update-frequency 2000
  --replay-start-size 30000
  --max-episode-steps 10000
  --train-per-step 2
  --frame-skip 4
  --linear-decay-steps 800000
  --n-step 4
  --per-alpha 0.6
  --per-beta-start 0.4
  --episodes 2000
```

- v4.4.1(RTX4080)

```bash
python dqn_v4.4.1.py 
  --env-name "ALE/Pong-v5"
  --wandb-project "DLP-Lab5-DQN-Pong(T3)-4080"
  --wandb-run-name "pong-enhanced-v4.4"
  --batch-size 512
  --memory-size 150000
  --lr 0.0.0005
  --epsilon-start 0.4
  --epsilon-decay 0.999
  --epsilon-min 0.1
  --target-update-frequency 500
  --replay-start-size 30000
  --max-episode-steps 10000
  --train-per-step 8
  --frame-skip 4
  --linear-decay-steps 100000
  --n-step 10
  --per-alpha 0.7
  --per-beta-start 0.5
  --episodes 1000
```

- v4.4(RTX4080) - 2

```bash
python dqn_v4.4.py 
  --env-name="ALE/Pong-v5"
  --wandb-project "DLP-Lab5-DQN-Pong(T3)-4080"
  --wandb-run-name "pong-enhanced-v4.4-max"
  --batch-size=256
  --memory-size=500000
  --lr=0.0003
  --discount-factor=0.99
  --epsilon-start=1.0
  --epsilon-decay=0.99999
  --epsilon-min=0.05
  --target-update-frequency=1500
  --replay-start-size=25000
  --max-episode-steps=10000
  --train-per-step=4
  --frame-skip=4
  --linear-decay-steps=600000
  --n-step=5
  --per-alpha=0.6
  --per-beta-start=0.4
  --episodes=1500
  --checkpoint-interval=25000
```


- v5

```bash
python dqn_v5.py \
  --env-name ALE/Pong-v5 \
  --wandb-project "DLP-Lab5-DQN-Pong(T3v5)-MPS" \
  --wandb-run-name "task3-enhanced-bs64" \
  --batch-size 64 \
  --memory-size 200000 \
  --replay-start-size 20000 \
  --linear-decay-steps 300000 \
  --epsilon-start 0.9 \
  --n-step 3 \
  --per-alpha 0.6 \
  --per-beta-start 0.4 
```

- v5(uint8)

```bash
python dqn_v5.py \
  --env-name ALE/Pong-v5 \
  --wandb-project "DLP-Lab5-DQN-Pong(T3v5)-MPS" \
  --wandb-run-name "task3-enhanced-uint8" \
  --batch-size 64 \
  --memory-size 200000 \
  --replay-start-size 20000 \
  --linear-decay-steps 300000 \
  --epsilon-start 0.9 \
  --n-step 3 \
  --per-alpha 0.6 \
  --per-beta-start 0.4 \
  --use-uint8
```

## Evaluation
### Task1

```bash
python evaluate_cartpole.py --model-path="./results/b7axzvnl/best_model.pt" --episodes=20
```

## Results
### Task1

- [b7axzvnl](https://wandb.ai/brend-main-nutc/DLP-Lab5-Task1/runs/b7axzvnl)


## 版本說明
- dqn_v2.py 完成了task1
- dqn_v3.py 嘗試了task2但有點失敗
- dqn_v4.py 嘗試task3中
- dqn_v4.2.py 基本完成task3要求 但效果不佳
- dqn_v4.3.py 接續4.2新增duel+noisy+drq(gpt)：失敗 完全學不到東西
- dqn_v4.3.py 接續4.2做微調(claude)
- dqn_v5.py 嘗試task3中 use-uint8