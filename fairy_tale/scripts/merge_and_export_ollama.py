"""
Ollama 모델 통합 및 내보내기 스크립트

QLoRA 어댑터를 베이스 모델과 병합하여 Ollama에서 사용할 수 있는 모델로 변환합니다.
"""

import os
import argparse
import shutil
from pathlib import Path
from loguru import logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from huggingface_hub import login
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers peft (or conda install -c conda-forge transformers)")


def merge_and_export(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    use_4bit: bool = False,
    hf_token: str | None = None,
):
    """
    LoRA 어댑터를 베이스 모델과 병합하여 내보내기
    
    Args:
        base_model_path: 베이스 모델 경로
        adapter_path: LoRA 어댑터 경로
        output_path: 출력 경로
        use_4bit: 4-bit 양자화 사용 여부 (병합 시에는 False 권장)
        hf_token: HuggingFace 토큰 (환경 변수에서도 자동으로 읽음)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Required packages not installed. Install with: "
            "pip install transformers peft (or conda install -c conda-forge transformers)"
        )
    
    # HuggingFace 토큰 확인 및 로그인
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        logger.info("Logging in to HuggingFace Hub...")
        login(token=token)
    elif not os.path.exists(base_model_path) and "/" in base_model_path:
        logger.warning("HuggingFace 모델 경로인데 토큰이 없습니다. 환경 변수 HF_TOKEN 또는 HUGGINGFACE_HUB_TOKEN을 설정하거나 --hf-token 옵션을 사용하세요.")
    
    logger.info("Merging LoRA adapters with base model...")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Adapter path: {adapter_path}")
    logger.info(f"Output path: {output_path}")
    
    # 토크나이저 로드
    logger.info("Loading tokenizer...")
    tokenizer_kwargs = {"use_fast": False}
    if token:
        tokenizer_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, **tokenizer_kwargs)
    
    # 베이스 모델 로드
    logger.info("Loading base model...")
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if token:
        model_kwargs["token"] = token
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    
    # LoRA 어댑터 로드 및 병합
    logger.info("Loading and merging LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    
    # 모델 저장
    logger.info(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    logger.success(f"Merged model saved to {output_path}")
    logger.info("You can now use this model with Ollama by creating a Modelfile")


def create_ollama_modelfile(
    model_path: str,
    character_name: str,
    output_path: str | None = None,
):
    """
    Ollama Modelfile 생성
    
    Args:
        model_path: 모델 경로
        character_name: 캐릭터 이름
        output_path: Modelfile 출력 경로 (기본값: model_path/Modelfile)
    """
    if output_path is None:
        output_path = os.path.join(model_path, "Modelfile")
    
    character_display_names = {
        "snow_white": "백설공주",
        "little_mermaid": "인어공주",
        "peter_pan": "피터펜",
        "three_pigs_little": "막내돼지"
    }
    
    display_name = character_display_names.get(character_name, character_name)
    
    # Modelfile은 모델 디렉토리 내에 생성되므로 상대 경로 사용
    # Ollama는 Modelfile이 있는 디렉토리를 기준으로 경로를 해석합니다
    modelfile_content = f"""FROM .

# {display_name} 캐릭터 Fine-tuning 모델
# Character: {character_name}
# Base model: Qwen/Qwen3-4B-Instruct-2507 (merged with LoRA adapters)

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER num_predict 300
PARAMETER repeat_penalty 1.1
PARAMETER repeat_last_n 64

# System prompt는 애플리케이션에서 설정
# MCP (Model Context Protocol)는 Agent 레벨에서 처리되므로 이 모델과 완전히 호환됩니다
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    
    logger.info(f"Created Modelfile at {output_path}")
    logger.info(f"To create Ollama model, run: ollama create {character_name} -f {output_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and export for Ollama")
    
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
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="LoRA 어댑터 경로 (기본값: fairy_tale/checkpoints/{character})"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="출력 경로 (기본값: fairy_tale/models/{character})"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="4-bit 양자화 사용 (병합 시에는 권장하지 않음)"
    )
    parser.add_argument(
        "--create_modelfile",
        action="store_true",
        help="Ollama Modelfile 생성"
    )
    
    args = parser.parse_args()
    
    # 기본 경로 설정
    base_dir = Path(__file__).parent.parent
    if args.adapter_path is None:
        args.adapter_path = str(base_dir / "checkpoints" / args.character)
    if args.output_path is None:
        args.output_path = str(base_dir / "models" / args.character)
    
    # 어댑터 경로 확인
    if not os.path.exists(args.adapter_path):
        logger.error(f"Adapter path not found: {args.adapter_path}")
        logger.info("Please run train_qlora.py first to train the model")
        return
    
    # 병합 및 내보내기
    merge_and_export(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        use_4bit=args.use_4bit,
    )
    
    # Modelfile 생성
    if args.create_modelfile:
        create_ollama_modelfile(
            model_path=args.output_path,
            character_name=args.character,
        )


if __name__ == "__main__":
    main()

