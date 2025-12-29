#!/bin/bash
# 백설공주 VTuber 설정 자동화 스크립트

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🎭 백설공주 VTuber 설정 시작..."

# 1. 캐릭터 전환
echo "📝 캐릭터 전환 중..."
python fairy_tale/switch_character.py snow_white

# 2. MCP 활성화
echo "🔧 MCP 활성화 중..."
sed -i "s/use_mcpp: False/use_mcpp: True/g" conf.yaml
sed -i 's/mcp_enabled_servers: \[\]/mcp_enabled_servers: ["time", "ddg-search"]/g' conf.yaml

# 3. 모델 변경
echo "🤖 모델 설정 변경 중..."
sed -i "s/model: 'qwen3:4b'/model: 'snow_white'/g" conf.yaml
sed -i "s/model: '.*'/model: 'snow_white'/g" conf.yaml 2>/dev/null || true
sed -i "s/temperature: 0.5/temperature: 0.7/g" conf.yaml
sed -i "s/temperature: [0-9.]\+/temperature: 0.7/g" conf.yaml 2>/dev/null || true

echo "✅ 설정 완료!"
echo ""
echo "🚀 서버 실행:"
echo "   python run_server.py"
echo ""
echo "📋 현재 설정 확인:"
echo "   python fairy_tale/switch_character.py --list"

