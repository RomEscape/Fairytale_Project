#!/bin/bash
# 백그라운드에서 대화 데이터 생성 스크립트
# Ctrl+C를 눌러도 프로세스가 계속 실행됨

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CHARACTER="${1:-snow_white}"
MODEL_NAME="${2:-exaone3.5:2.4b}"
DATA_PATH="${3:-fairy_tale/processed/2025-12-10/generated_agent_scene_snow_white-korean.json}"

LOG_FILE="dialogue_generation_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="dialogue_generation.pid"

echo "=================================================================================="
echo "백그라운드 대화 데이터 생성 시작"
echo "=================================================================================="
echo "캐릭터: $CHARACTER"
echo "모델: $MODEL_NAME"
echo "장면 데이터: $DATA_PATH"
echo "로그 파일: $LOG_FILE"
echo "PID 파일: $PID_FILE"
echo "=================================================================================="
echo ""

# 기존 프로세스 확인
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "경고: 기존 프로세스가 실행 중입니다 (PID: $OLD_PID)"
        echo "종료하려면: kill $OLD_PID"
        exit 1
    else
        echo "기존 PID 파일 삭제 중..."
        rm -f "$PID_FILE"
    fi
fi

# nohup으로 백그라운드 실행
nohup python fairy_tale/run_gen_data.py \
    --character "$CHARACTER" \
    --prompt-name gen_dialogue \
    --model_name "$MODEL_NAME" \
    --data-path "$DATA_PATH" \
    > "$LOG_FILE" 2>&1 &

# PID 저장
echo $! > "$PID_FILE"
PID=$(cat "$PID_FILE")

echo "백그라운드 프로세스 시작됨 (PID: $PID)"
echo ""
echo "명령어:"
echo "  로그 확인: tail -f $LOG_FILE"
echo "  프로세스 확인: ps aux | grep $PID"
echo "  프로세스 종료: kill $PID"
echo "  또는: pkill -f 'run_gen_data.py.*gen_dialogue'"
echo ""
echo "로그 파일: $LOG_FILE"
echo "PID 파일: $PID_FILE"
echo ""
echo "프로세스는 백그라운드에서 계속 실행됩니다."
echo "터미널을 닫거나 Ctrl+C를 눌러도 프로세스는 계속 실행됩니다."

