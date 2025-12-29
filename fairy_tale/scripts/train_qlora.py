"""
QLoRA Fine-tuning 스크립트

Qwen3:4b 모델을 기반으로 QLoRA Fine-tuning을 수행합니다.
참고 프로젝트의 train_sumi_jo.py 구조를 따릅니다.
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Optional
from loguru import logger

# Wandb 설정
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")

try:
    from huggingface_hub import login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Install with: pip install huggingface_hub")

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Transformers not available: {e}. Install with: pip install transformers peft bitsandbytes (or conda install -c conda-forge transformers)")


# 커스텀 DataCollator 제거 (trainable-agents 방식에서는 불필요)
# DataCollatorForLanguageModeling이 자동으로 labels 생성하므로 커스텀 클래스 불필요


def load_jsonl_dataset(tokenizer, data_path: str, model_max_length: int = 2048):
    """
    JSONL 데이터셋 로드 (trainable-agents 방식)
    
    각 라인은 {"prompt": "...", "output": "...", "source": "..."} 형식
    trainable-agents와 동일하게 prompt와 output을 단순 연결하여 전체를 학습합니다.
    
    Args:
        tokenizer: 토크나이저
        data_path: JSONL 파일 경로
        model_max_length: 최대 시퀀스 길이
    
    Returns:
        데이터셋 객체 (input_ids만 포함, labels는 DataCollator가 자동 생성)
    """
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception as e:
                logger.warning(f"Failed to parse line: {e}")
                continue
            
            prompt = ex.get('prompt', '')
            output = ex.get('output', '')
            
            if not prompt or not output:
                continue
            
            # trainable-agents 방식: 단순 연결 (chat template 미사용)
            text = f"{prompt}{output}"
            texts.append(text)
    
    if not texts:
        raise ValueError(f"No valid data found in {data_path}")
    
    logger.info(f"Loaded {len(texts)} samples from {data_path}")
    logger.info("Using trainable-agents style: simple concatenation (no chat template, no loss masking)")
    
    # 전체 텍스트를 토크나이징
    encodings = tokenizer(
        texts,
        max_length=model_max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    
    # input_ids만 반환 (labels 없음, DataCollator가 자동 생성)
    dataset = encodings['input_ids']
    
    class JsonlDataset:
        """JSONL 데이터셋 래퍼 (trainable-agents 방식)"""
        def __init__(self, input_ids_list):
            self.input_ids_list = input_ids_list
        
        def __len__(self):
            return len(self.input_ids_list)
        
        def __getitem__(self, idx):
            ids = self.input_ids_list[idx]
            return {
                'input_ids': ids,  # labels 없음 (DataCollator가 자동 생성)
            }
    
    return JsonlDataset(dataset)


def train_model_qlora(
    base_model_path: str,
    output_dir: str,
    data_path: str,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    model_max_length: int = 2048,
    save_steps: int = 100,
    logging_steps: int = 10,
    hf_token: str | None = None,
):
    """
    QLoRA (4-bit quantization + LoRA adapters) Fine-tuning
    
    Args:
        base_model_path: 베이스 모델 경로 (HuggingFace 모델명 또는 로컬 경로)
        output_dir: 출력 디렉토리 (LoRA 어댑터 저장 위치)
        data_path: 학습 데이터 경로 (JSONL 파일)
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: 학습률
        num_train_epochs: 에폭 수
        per_device_train_batch_size: 디바이스당 배치 크기
        gradient_accumulation_steps: 그래디언트 누적 스텝
        model_max_length: 최대 시퀀스 길이
        save_steps: 저장 스텝 간격
        logging_steps: 로깅 스텝 간격
        hf_token: HuggingFace 토큰 (환경 변수 HF_TOKEN 또는 HUGGINGFACE_HUB_TOKEN도 사용 가능)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Required packages not installed. Install with: "
            "pip install transformers peft bitsandbytes accelerate (or conda install -c conda-forge transformers)"
        )
    
    logger.info("Starting QLoRA training...")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Output directory (adapters): {output_dir}")
    logger.info(f"Training data: {data_path}")
    logger.info(f"LoRA r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Wandb 초기화
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key and WANDB_AVAILABLE:
        # 프로젝트 이름과 run 이름 설정
        project_name = "dreamtale"
        run_name = f"snow_white-qlora-{Path(data_path).stem}"
        
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "base_model": base_model_path,
                "output_dir": output_dir,
                "data_path": data_path,
                "lora_r": r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
                "num_train_epochs": num_train_epochs,
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "model_max_length": model_max_length,
                "save_steps": save_steps,
                "logging_steps": logging_steps,
            }
        )
        logger.info(f"Wandb initialized: project={project_name}, run={run_name}")
    elif wandb_api_key and not WANDB_AVAILABLE:
        logger.warning("WANDB_API_KEY is set but wandb is not installed. Install with: pip install wandb")
    else:
        logger.info("Wandb not initialized (WANDB_API_KEY not set)")
    
    # HuggingFace 토큰 인증
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    if hf_token and HF_HUB_AVAILABLE:
        try:
            login(token=hf_token)
            logger.info("Logged in to HuggingFace Hub.")
        except Exception as e:
            logger.warning(f"Failed to login to HuggingFace Hub: {e}")
    elif hf_token:
        logger.warning("HuggingFace token provided but huggingface_hub not available. Some models might not be accessible.")
    
    # BitsAndBytesConfig 설정 (4-bit 양자화)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # 토크나이저 로드
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # 모델 로드 (4-bit 양자화)
    logger.info("Loading 4-bit quantized base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    
    # k-bit 학습 준비
    try:
        from peft import prepare_model_for_kbit_training as _prepare_model_for_kbit_training
    except Exception:
        try:
            from peft.utils.other import prepare_model_for_kbit_training as _prepare_model_for_kbit_training
        except Exception as e:
            raise ImportError(
                "prepare_model_for_kbit_training not found. Please install/update peft: "
                "pip install -U peft (or conda install -c conda-forge peft)"
            ) from e
    
    model.gradient_checkpointing_enable()
    model = _prepare_model_for_kbit_training(model)
    
    # LoRA 설정
    # Qwen 모델은 Qwen 타겟 모듈 사용
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, lora_config)
    
    logger.info("Trainable parameters:")
    model.print_trainable_parameters()
    
    # 데이터셋 로드
    logger.info("Tokenizing dataset...")
    from tqdm import tqdm
    train_dataset = load_jsonl_dataset(tokenizer, data_path, model_max_length=model_max_length)
    
    total_samples = len(train_dataset)
    total_steps = (total_samples // (per_device_train_batch_size * gradient_accumulation_steps)) * num_train_epochs
    logger.info(f"Total training samples: {total_samples}")
    logger.info(f"Estimated total steps: {total_steps}")
    logger.info(f"Training progress will be displayed with tqdm")
    
    # Data collator 설정 (trainable-agents 방식: 기본 DataCollator 사용)
    # DataCollatorForLanguageModeling이 자동으로 labels 생성 (input_ids를 그대로 복사)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments (trainable-agents와 동일한 설정)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.03,  # trainable-agents와 동일
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,  # trainable-agents와 동일 (logging_steps 파라미터 무시)
        save_strategy="epoch",  # trainable-agents와 동일 ("steps" → "epoch")
        save_total_limit=3,  # trainable-agents와 동일
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        fp16=False,
        optim="paged_adamw_32bit",  # trainable-agents와 동일
        gradient_checkpointing=True,  # 메모리 절약
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,  # 메모리 절약
        report_to=["wandb"] if (os.getenv("WANDB_API_KEY") and WANDB_AVAILABLE) else [],  # wandb API 키가 있으면 활성화
        run_name=f"snow_white-qlora-{Path(data_path).stem}" if (os.getenv("WANDB_API_KEY") and WANDB_AVAILABLE) else None,
        remove_unused_columns=False,
        disable_tqdm=False,  # tqdm 진행 표시 활성화
    )
    
    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 학습 시작
    logger.info("Starting training...")
    trainer.train()
    
    # LoRA 어댑터 및 토크나이저 저장
    logger.info(f"Saving LoRA adapters and tokenizer to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.success(f"Training completed! Adapters saved to {output_dir}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for Fairy Tale Characters")
    
    # 필수 인자
    parser.add_argument(
        "--character",
        type=str,
        required=True,
        choices=["snow_white", "little_mermaid", "peter_pan", "three_pigs_little"],
        help="캐릭터 이름"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-4B-Instruct",
        help="베이스 모델 경로 (HuggingFace 모델명 또는 로컬 경로)"
    )
    
    # 출력 디렉토리
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="출력 디렉토리 (기본값: fairy_tale/checkpoints/{character})"
    )
    
    # 데이터 경로
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="학습 데이터 경로 (기본값: fairy_tale/training_data/{character}/training_data.jsonl)"
    )
    
    # LoRA 하이퍼파라미터
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    # 기타
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    
    args = parser.parse_args()
    
    # 기본 경로 설정
    base_dir = Path(__file__).parent.parent
    if args.output_dir is None:
        args.output_dir = str(base_dir / "checkpoints" / args.character)
    if args.data_path is None:
        args.data_path = str(base_dir / "training_data" / args.character / "training_data.jsonl")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 데이터 파일 확인
    if not os.path.exists(args.data_path):
        logger.error(f"Training data not found: {args.data_path}")
        logger.info("Please run convert_to_jsonl.py first to generate JSONL data")
        return
    
    # 학습 실행
    train_model_qlora(
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        data_path=args.data_path,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        model_max_length=args.max_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()

