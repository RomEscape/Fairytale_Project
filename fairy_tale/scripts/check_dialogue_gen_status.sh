#!/bin/bash
# 대화 데이터 생성 프로세스 상태 확인 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PID_FILE="dialogue_generation.pid"

echo "=================================================================================="
echo "대화 데이터 생성 프로세스 상태 확인"
echo "=================================================================================="

if [ ! -f "$PID_FILE" ]; then
    echo "오류: PID 파일이 없습니다. 프로세스가 실행 중이지 않을 수 있습니다."
    echo ""
    echo "실행 중인 프로세스 확인:"
    ps aux | grep "run_gen_data.py.*gen_dialogue" | grep -v grep || echo "  실행 중인 프로세스 없음"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "프로세스 실행 중 (PID: $PID)"
    echo ""
    echo "프로세스 정보:"
    ps -p "$PID" -o pid,ppid,cmd,etime,%mem,%cpu
    echo ""
    
    # 로그 파일 찾기
    LOG_FILES=$(ls -t dialogue_generation_*.log 2>/dev/null | head -1)
    if [ -n "$LOG_FILES" ]; then
        echo "최근 로그 파일: $LOG_FILES"
        echo ""
        echo "마지막 20줄:"
        tail -20 "$LOG_FILES"
    else
        echo "로그 파일을 찾을 수 없습니다."
    fi
else
    echo "오류: 프로세스가 실행 중이지 않습니다 (PID: $PID)"
    echo "PID 파일 삭제 중..."
    rm -f "$PID_FILE"
    echo ""
    echo "실행 중인 프로세스 확인:"
    ps aux | grep "run_gen_data.py.*gen_dialogue" | grep -v grep || echo "  실행 중인 프로세스 없음"
fi

