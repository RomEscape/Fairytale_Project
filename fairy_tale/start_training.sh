#!/bin/bash

# 스크립트 디렉토리로 이동 (상대 경로 사용)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 환경 변수 설정 (토큰은 환경변수에서 설정하거나 스크립트 실행 전에 설정하세요)
# export WANDB_API_KEY='your_wandb_api_key_here'
# export HF_TOKEN='your_huggingface_token_here'

# 로그 파일명 생성
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

echo "학습을 시작합니다..."
echo "로그 파일: $LOG_FILE"
echo "진행 상황 확인: tail -f $LOG_FILE"

# nohup으로 백그라운드 실행
nohup python train_character.py \
    --character snow_white \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --data-path processed/2025-12-26/prompted/prompted_agent_dialogue_snow_white.jsonl \
    --output-dir checkpoints/snow_white \
    --qlora \
    --epochs 3 \
    --learning-rate 2e-4 \
    --batch-size 2 \
    --grad-accum 8 \
    > "$LOG_FILE" 2>&1 &

# 프로세스 ID 저장
TRAIN_PID=$!
echo "학습 프로세스 ID: $TRAIN_PID"
echo "$TRAIN_PID" > training.pid
echo ""
echo "학습이 백그라운드에서 시작되었습니다."
echo "Ctrl+C를 눌러도 학습은 계속 진행됩니다."
echo ""
echo "진행 상황 확인: tail -f $LOG_FILE"
echo "프로세스 확인: ps aux | grep $TRAIN_PID"
echo "학습 중지: kill $TRAIN_PID"
