#!/bin/bash
MAX_ATTEMPTS=10
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "啟動訓練 (嘗試 $ATTEMPT/$MAX_ATTEMPTS)"
    
    python dqn_v3.py --env-name "ALE/Pong-v5" \
        --wandb-project "DLP-Lab5-DQN-Pong(T2)-MPS(auto)" \
        --wandb-run-name "pong-robust" \
        --wandb-id "rh07648q" \
        --batch-size 32 \
        --memory-size 200000 \
        --replay-start-size 5000 \
        --lr 0.00025 \
        --epsilon-decay 0.9999 \
        --target-update-frequency 1000 \
        --checkpoint-interval 25
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "訓練成功完成！"
        break
    else
        echo "訓練中斷，退出代碼: $EXIT_CODE. 嘗試重新啟動..."
        sleep 5
        ATTEMPT=$((ATTEMPT+1))
    fi
done