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
    --memory-size 100000 \
    --lr 0.00025 \
    --linear-decay-steps 1000000 \
    --frame-skip 4 \
    --train-per-step 2 \
    --target-update-frequency 10000 \
    --replay-start-size 5000 \
    --episodes 4000
```

## Evaluation
### Task1

```bash
python evaluate_cartpole.py --model-path="./results/b7axzvnl/best_model.pt" --episodes=20
```

## Results
### Task1

- [b7axzvnl](https://wandb.ai/brend-main-nutc/DLP-Lab5-Task1/runs/b7axzvnl)
