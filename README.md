# 동화 기반 페르소나 LLM 시스템

동화 캐릭터의 페르소나를 학습한 대규모 언어 모델(LLM)을 Fine-tuning하여 구현한 AI 캐릭터 시스템입니다.

이 프로젝트는 동화 속 캐릭터(예: 백설공주)의 성격, 말투, 배경 스토리를 학습 데이터로 생성하고, QLoRA 기반 Fine-tuning을 통해 해당 캐릭터의 페르소나를 모델에 주입합니다. Fine-tuning된 모델은 Ollama를 통해 추론되며, Open-LLM-VTuber 프레임워크와 통합되어 음성 대화 및 Live2D 아바타와 함께 동작합니다.

## 주요 특징

- **동화 캐릭터 페르소나 학습**: 동화 스토리 기반 장면 및 대화 데이터 생성
- **QLoRA Fine-tuning**: 효율적인 파라미터 효율 Fine-tuning 기법 사용
- **Ollama 통합**: GGUF 형식으로 변환하여 Ollama에서 추론
- **VTuber 시스템 연동**: Open-LLM-VTuber와 통합하여 음성 대화 및 Live2D 지원

## 빠른 시작 (Quick Start)

**이미 학습된 백설공주 모델을 바로 실행하려면:**

### 1. 저장소 클론

```bash
git clone https://github.com/RomEscape/Fairytale_Project.git
cd Fairytale_Project
```

### 2. 의존성 설치

```bash
# conda 사용자
pip install -r requirements.txt

# 또는 uv 사용자
uv sync
```

### 3. 모델 파일 다운로드 및 Ollama 설치

**주의**: 모델 파일(2.4GB)은 GitHub LFS 크기 제한(2GB)으로 인해 저장소에 포함되지 않습니다. HuggingFace에서 다운로드하세요.

```bash
# Ollama 설치 (아직 설치하지 않은 경우)
curl -fsSL https://ollama.com/install.sh | sh

# 모델 디렉토리 생성
mkdir -p fairy_tale/models/snow_white_gguf

# HuggingFace Hub 설치 및 로그인 (처음만)
pip install huggingface_hub
huggingface-cli login  # HuggingFace 토큰 입력

# HuggingFace에서 모델 다운로드
huggingface-cli download PJiNH/snow_white_gguf model-q4_0.gguf \
    --local-dir fairy_tale/models/snow_white_gguf/

# 모델 확인: https://huggingface.co/PJiNH/snow_white_gguf

# Modelfile 확인/생성
cd fairy_tale/models/snow_white_gguf
if [ ! -f Modelfile ]; then
    cat > Modelfile << 'EOF'
FROM model-q4_0.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER num_predict 300
PARAMETER repeat_penalty 1.1
PARAMETER repeat_last_n 64
EOF
fi

# Ollama에 모델 등록
ollama rm snow_white 2>/dev/null || true
ollama create snow_white -f Modelfile

# 모델 등록 확인
ollama list | grep snow_white
```

### 4. VTuber 설정 및 서버 실행

```bash
# 프로젝트 루트로 돌아가기
cd ../..

# 백설공주 캐릭터 설정 적용
./fairy_tale/setup_snow_white.sh

# 서버 실행
python run_server.py
```

### 5. 브라우저에서 접속

1. 웹 브라우저에서 `http://localhost:12393` 접속
2. 마이크 권한 허용 (음성 대화를 위해)
3. 백설공주와 대화 시작!

## 지원 모델

- **베이스 모델**: Qwen/Qwen3-4B-Instruct-2507 (HuggingFace)
- **데이터 생성 모델**: exaone3.5:7.8b (Ollama, 한국어 특화)
- **Fine-tuning 모델**: snow_white (Ollama에 등록된 Fine-tuning 모델)

## 워크플로우

```
프로파일 → 장면 생성 → 대화 생성 → 파싱 → FT 데이터 준비 → QLoRA Fine-tuning → 모델 병합 → GGUF 변환 → Ollama 등록 → VTuber 설정
```

## 사전 준비

```bash
# 1. Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 2. 데이터 생성용 모델 다운로드 (권장: 7.8B)
ollama pull exaone3.5:7.8b
```

## 단계별 실행

**주의**: 모든 명령어는 프로젝트 루트 디렉토리에서 실행합니다.

### 1단계: 장면 데이터 생성

```bash
python fairy_tale/run_gen_data.py --character snow_white --prompt-name gen_scene --model_name exaone3.5:7.8b
```

**출력**: `fairy_tale/result/{date}/gen_scene/scene_*.jsonl`

### 2단계: 장면 데이터 파싱

```bash
python fairy_tale/parser/parse_data_scene.py fairy_tale/result/{date}/gen_scene/scene_*.jsonl
```

**출력**: `fairy_tale/processed/{date}/generated_agent_scene_snow_white-korean.json`

### 3단계: 대화 데이터 생성

```bash
# 백그라운드 실행 (권장)
./fairy_tale/scripts/run_dialogue_gen_background.sh

# 또는 직접 실행
python fairy_tale/run_gen_data.py --character snow_white --prompt-name gen_dialogue \
    --model_name exaone3.5:7.8b \
    --data-path fairy_tale/processed/{date}/generated_agent_scene_snow_white-korean.json
```

**출력**: `fairy_tale/result/{date}/gen_dialogue/dialogue_*.jsonl`

### 4단계: 대화 데이터 파싱

```bash
python fairy_tale/parser/parse_data_dialogue.py fairy_tale/result/{date}/gen_dialogue/dialogue_*.jsonl
```

**출력**: `fairy_tale/processed/{date}/generated_agent_dialogue_snow_white-korean.json`

### 5단계: Fine-tuning 데이터 준비

```bash
python fairy_tale/parser/convert_prompt_data.py fairy_tale/processed/{date}/generated_agent_dialogue_snow_white-korean.json
```

**출력**: `fairy_tale/processed/{date}/prompted/prompted_agent_dialogue_snow_white.jsonl`

### 6단계: 학습 데이터 품질 확인 (권장)

```bash
python fairy_tale/scripts/check_training_data_quality.py \
    fairy_tale/processed/{date}/prompted/prompted_agent_dialogue_snow_white.jsonl
```

**목표**: 백설공주 언급 비율 50% 이상

### 7단계: QLoRA Fine-tuning

```bash
# Wandb 설정 (선택사항)
export WANDB_API_KEY='your_wandb_api_key_here'

# Fine-tuning 실행
python fairy_tale/scripts/train_qlora.py \
    --character snow_white \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --data_path fairy_tale/processed/{date}/prompted/prompted_agent_dialogue_snow_white.jsonl \
    --output_dir fairy_tale/checkpoints/snow_white \
    --epochs 3 \
    --learning_rate 2e-4 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_length 2048
```

**출력**: `fairy_tale/checkpoints/snow_white/`

### 8단계: 모델 병합

```bash
cd fairy_tale
export HF_TOKEN=your_huggingface_token_here

python scripts/merge_and_export_ollama.py \
    --character snow_white \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter_path checkpoints/snow_white \
    --output_path models/snow_white \
    --create_modelfile
```

**출력**: `fairy_tale/models/snow_white/`

### 9단계: GGUF 변환 및 Ollama 등록

**주의**: Qwen3 모델은 GGUF 변환이 필수입니다.

```bash
# 상태 확인
ollama list | grep snow_white

# 이미 등록되어 있으면 10단계로 진행
```

**아직 완료하지 않은 경우**:

```bash
# 1. llama.cpp 설치 (한 번만)
[ ! -d "llama.cpp" ] && git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --config Release

# 2. GGUF 변환
cd ..
mkdir -p fairy_tale/models/snow_white_gguf
cd llama.cpp
[ ! -d "venv" ] && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && deactivate

source venv/bin/activate
python convert_hf_to_gguf.py \
    ../fairy_tale/models/snow_white \
    --outfile ../fairy_tale/models/snow_white_gguf/model-f16.gguf \
    --outtype f16
deactivate

# 양자화
./build/bin/llama-quantize \
    ../fairy_tale/models/snow_white_gguf/model-f16.gguf \
    ../fairy_tale/models/snow_white_gguf/model-q4_0.gguf \
    q4_0

# 3. Modelfile 생성 및 Ollama 등록
cd ../fairy_tale/models/snow_white_gguf
cat > Modelfile << 'EOF'
FROM model-q4_0.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER num_predict 300
PARAMETER repeat_penalty 1.1
PARAMETER repeat_last_n 64
EOF

ollama rm snow_white 2>/dev/null || true
ollama create snow_white -f Modelfile
```

### 10단계: VTuber 설정

```bash
# 자동 설정 (권장)
./fairy_tale/setup_snow_white.sh

# 또는 수동 설정
python fairy_tale/switch_character.py snow_white
sed -i "s/use_mcpp: False/use_mcpp: True/g" conf.yaml
sed -i 's/mcp_enabled_servers: \[\]/mcp_enabled_servers: ["time", "ddg-search"]/g" conf.yaml
sed -i "s/model: 'qwen3:4b'/model: 'snow_white'/g" conf.yaml
sed -i "s/temperature: 0.5/temperature: 0.7/g" conf.yaml

# 서버 실행
python run_server.py
```

**대화 시작**:
1. 서버 실행 후 웹 브라우저에서 `http://localhost:12393` 접속
2. 마이크 권한 허용 (음성 대화를 위해)
3. 대화 시작!

## 모델 구분

| 용도 | 모델 | 설명 |
|------|------|------|
| 데이터 생성 | `exaone3.5:7.8b` | Ollama, 한국어 특화 (권장) |
| Fine-tuning | `Qwen/Qwen3-4B-Instruct-2507` | HuggingFace |
| 추론 | `snow_white` | Ollama에 등록된 Fine-tuning 모델 |

## 디렉토리 구조

```
fairy_tale/
├── data/seed_data/          # 프로파일 및 프롬프트
├── parser/                   # 파싱 스크립트
├── scripts/                  # 유틸리티 스크립트
├── result/{date}/            # 생성된 데이터
├── processed/{date}/          # 파싱된 데이터
├── checkpoints/{character}/  # QLoRA 어댑터
└── models/{character}/       # 병합된 모델
```

## 학습 설정 권장사항

- **에폭**: 3 (trainable-agents와 동일)
- **학습률**: 2e-4 (trainable-agents와 동일)
- **배치 크기**: 메모리에 따라 조정 (기본: 4)
- **최대 길이**: 2048 (메모리 부족 시 1024로 감소)

## 데이터 생성 권장사항

- **모델**: `exaone3.5:7.8b` 사용 권장 (안정적인 생성)
- **장면당 대화**: 23개 (조수미 데이터 수준)
- **목표 데이터**: 40,000개 이상
