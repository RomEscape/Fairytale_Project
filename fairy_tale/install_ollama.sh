#!/bin/bash
# Ollama 및 Qwen3 4B 모델 설치 스크립트
# 리눅스 환경에서 Ollama를 설치하고 Qwen3 4B 모델을 다운로드합니다.

set -e  # 오류 발생 시 스크립트 중단

echo "=========================================="
echo "Ollama 및 Qwen3 4B 모델 설치 스크립트"
echo "=========================================="
echo ""

# 1. Ollama 설치 확인
if command -v ollama &> /dev/null; then
    echo "✅ Ollama가 이미 설치되어 있습니다."
    OLLAMA_VERSION=$(ollama --version)
    echo "   버전: $OLLAMA_VERSION"
else
    echo "📦 Ollama를 설치합니다..."
    echo "   (sudo 권한이 필요합니다)"
    curl -fsSL https://ollama.com/install.sh | sh
    
    if command -v ollama &> /dev/null; then
        echo "✅ Ollama 설치가 완료되었습니다."
    else
        echo "❌ Ollama 설치에 실패했습니다. 수동으로 설치해주세요."
        echo "   참고: https://ollama.com/download"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Qwen3 4B 모델 다운로드"
echo "=========================================="
echo ""

# 2. Ollama 서비스 확인 및 시작
echo "🔍 Ollama 서비스 상태 확인 중..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama 서비스가 실행되고 있지 않습니다. 서비스를 시작합니다..."
    
    # systemd를 사용하는 경우 시도
    if command -v systemctl &> /dev/null && systemctl is-system-running > /dev/null 2>&1; then
        echo "   systemd를 통해 서비스를 시작합니다..."
        sudo systemctl start ollama 2>/dev/null || echo "   systemd로 시작 실패, 다른 방법 시도..."
        sleep 2
    fi
    
    # 서비스가 여전히 실행되지 않으면 직접 ollama serve 실행
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "   ollama serve를 백그라운드로 직접 시작합니다..."
        # 기존 프로세스 확인
        if pgrep -f "ollama serve" > /dev/null; then
            echo "   ✅ ollama serve 프로세스가 이미 실행 중입니다."
        else
            # 백그라운드로 ollama serve 시작
            nohup ollama serve > /dev/null 2>&1 &
            echo "   ⏳ Ollama 서비스 시작 대기 중... (최대 10초)"
            
            # 최대 10초 동안 서비스 시작 대기
            for i in {1..10}; do
                sleep 1
                if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                    break
                fi
            done
        fi
    fi
    
    # 서비스 상태 최종 확인
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo ""
        echo "❌ Ollama 서비스를 자동으로 시작할 수 없습니다."
        echo ""
        echo "다음 명령어로 수동으로 서비스를 시작해주세요:"
        echo "   ollama serve"
        echo ""
        echo "또는 다른 터미널에서 실행한 후, 이 스크립트를 다시 실행하세요."
        exit 1
    fi
fi

echo "✅ Ollama 서비스가 실행 중입니다."

# 3. Qwen3 4B 모델 확인 및 다운로드
echo ""
echo "🔍 Qwen3 4B 모델 확인 중..."

MODEL_EXISTS=false
if command -v ollama &> /dev/null; then
    if ollama list | grep -q "qwen3:4b"; then
        MODEL_EXISTS=true
        echo "✅ Qwen3 4B 모델이 이미 다운로드되어 있습니다."
    fi
fi

if [ "$MODEL_EXISTS" = false ]; then
    echo "📥 Qwen3 4B 모델을 다운로드합니다..."
    echo "   (약 2.6GB, 네트워크 속도에 따라 시간이 걸릴 수 있습니다)"
    echo ""
    
    ollama pull qwen3:4b
    
    if ollama list | grep -q "qwen3:4b"; then
        echo "✅ Qwen3 4B 모델 다운로드가 완료되었습니다."
    else
        echo "❌ Qwen3 4B 모델 다운로드에 실패했습니다."
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "설치 완료"
echo "=========================================="
echo ""
echo "다운로드된 모델 목록:"
ollama list
echo ""
echo "✅ 모든 설치가 완료되었습니다!"
echo ""
echo "다음 단계:"
echo "1. 모델 테스트: ollama run qwen3:4b"
echo "2. VTuber 설정 파일(conf.yaml)에서 qwen3:4b 모델 사용 설정"
echo ""

