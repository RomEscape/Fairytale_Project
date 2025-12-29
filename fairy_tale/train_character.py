"""
캐릭터 Fine-tuning 통합 스크립트

QLoRA Fine-tuning을 수행하고 Ollama 모델로 내보냅니다.
참고 프로젝트의 train_sumi_jo.py 구조를 따릅니다.
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="캐릭터 Fine-tuning")
    parser.add_argument(
        "--character",
        type=str,
        required=True,
        choices=["snow_white", "little_mermaid", "peter_pan", "three_pigs_little"],
        help="캐릭터 이름"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="베이스 모델 경로 (HuggingFace 모델명 또는 로컬 경로)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="학습 데이터 경로 (기본값: processed/{date}/prompted/prompted_agent_dialogue_{character}.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="출력 디렉토리 (기본값: checkpoints/{character})"
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="QLoRA 학습 사용"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="모델 병합 및 Ollama 내보내기"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="에폭 수"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="배치 크기"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="그래디언트 누적 스텝"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="학습률 (기본값: 2e-4)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace 토큰 (환경 변수 HF_TOKEN 또는 HUGGINGFACE_HUB_TOKEN도 사용 가능)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    
    # 기본 경로 설정
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d')
    
    if args.data_path is None:
        args.data_path = str(base_dir / "processed" / now / "prompted" / f"prompted_agent_dialogue_{args.character}.jsonl")
    
    if args.output_dir is None:
        args.output_dir = str(base_dir / "checkpoints" / args.character)
    
    # QLoRA 학습
    if args.qlora:
        # 데이터 경로 확인 (훈련 시에만 필요)
        if not os.path.exists(args.data_path):
            logger.error(f"Training data not found: {args.data_path}")
            logger.info("Please run run_gen_data.py first to generate training data")
            return 1
        logger.info("Starting QLoRA training...")
        # train_qlora 모듈을 직접 import
        sys.path.insert(0, str(base_dir / "scripts"))
        from train_qlora import train_model_qlora
        
        # HuggingFace 토큰 확인
        hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        
        train_model_qlora(
            base_model_path=args.base_model,
            output_dir=args.output_dir,
            data_path=args.data_path,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            hf_token=hf_token,
        )
    
    # 모델 병합 및 Ollama 내보내기
    if args.merge:
        # 어댑터 경로 확인 (병합 시 필요)
        if not os.path.exists(args.output_dir):
            logger.error(f"Adapter path not found: {args.output_dir}")
            logger.info("Please run training first with --qlora flag")
            return 1
        
        logger.info("Merging and exporting for Ollama...")
        # merge_and_export_ollama 모듈을 직접 import
        sys.path.insert(0, str(base_dir / "scripts"))
        from merge_and_export_ollama import merge_and_export, create_ollama_modelfile
        
        model_output_dir = str(base_dir / "models" / args.character)
        
        # HuggingFace 토큰 확인
        hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        
        merge_and_export(
            base_model_path=args.base_model,
            adapter_path=args.output_dir,
            output_path=model_output_dir,
            use_4bit=False,
            hf_token=hf_token,
        )
        
        create_ollama_modelfile(
            model_path=model_output_dir,
            character_name=args.character,
        )
        
        logger.success(f"Model exported to {model_output_dir}")
        logger.info(f"To create Ollama model, run: ollama create {args.character} -f {model_output_dir}/Modelfile")
    
    logger.success("Training completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

