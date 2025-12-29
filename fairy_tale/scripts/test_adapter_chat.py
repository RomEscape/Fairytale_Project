"""
LoRA 어댑터 대화 테스트 스크립트

학습된 LoRA 어댑터를 로드하여 간단한 대화를 테스트합니다.
"""

import os
import argparse
from pathlib import Path
from loguru import logger

try:
 import torch
 from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
 from peft import PeftModel
 TRANSFORMERS_AVAILABLE = True
except ImportError as e:
 TRANSFORMERS_AVAILABLE = False
 logger.error(f"Required packages not installed: {e}")
 logger.error("Install with: pip install transformers peft bitsandbytes accelerate")

try:
 from huggingface_hub import login
 HF_HUB_AVAILABLE = True
except ImportError:
 HF_HUB_AVAILABLE = False
 logger.warning("huggingface_hub not available")

def load_model_with_adapter(
 base_model_path: str,
 adapter_path: str,
 hf_token: str | None = None,
 use_4bit: bool = True,
):
 """
 베이스 모델과 LoRA 어댑터를 로드합니다.
 
 Args:
 base_model_path: 베이스 모델 경로
 adapter_path: LoRA 어댑터 경로
 hf_token: HuggingFace 토큰
 use_4bit: 4-bit 양자화 사용 여부
 
 Returns:
 (model, tokenizer) 튜플
 """
 if not TRANSFORMERS_AVAILABLE:
 raise ImportError("Required packages not installed")
 
 # HuggingFace 토큰 인증
 if hf_token is None:
 hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
 
 if hf_token and HF_HUB_AVAILABLE:
 try:
 login(token=hf_token)
 logger.info("Logged in to HuggingFace Hub.")
 except Exception as e:
 logger.warning(f"Failed to login to HuggingFace Hub: {e}")
 
 # 토크나이저 로드
 logger.info(f"Loading tokenizer from {base_model_path}...")
 tokenizer = AutoTokenizer.from_pretrained(
 base_model_path,
 use_fast=False,
 token=hf_token,
 trust_remote_code=True,
 )
 if tokenizer.pad_token is None:
 tokenizer.pad_token = tokenizer.eos_token
 
 # 모델 로드
 if use_4bit:
 logger.info("Loading 4-bit quantized base model...")
 bnb_config = BitsAndBytesConfig(
 load_in_4bit=True,
 bnb_4bit_quant_type="nf4",
 bnb_4bit_use_double_quant=True,
 bnb_4bit_compute_dtype=torch.bfloat16,
 )
 model = AutoModelForCausalLM.from_pretrained(
 base_model_path,
 quantization_config=bnb_config,
 device_map="auto",
 trust_remote_code=True,
 token=hf_token,
 )
 else:
 logger.info("Loading full precision base model...")
 model = AutoModelForCausalLM.from_pretrained(
 base_model_path,
 device_map="auto",
 trust_remote_code=True,
 token=hf_token,
 torch_dtype=torch.bfloat16,
 )
 
 # LoRA 어댑터 로드
 logger.info(f"Loading LoRA adapter from {adapter_path}...")
 model = PeftModel.from_pretrained(model, adapter_path)
 model.eval()
 
 logger.info("Model and adapter loaded successfully!")
 return model, tokenizer

def chat(
 model,
 tokenizer,
 user_input: str,
 max_new_tokens: int = 512,
 temperature: float = 0.7,
 top_p: float = 0.9,
):
 """
 모델과 대화를 수행합니다.
 
 Args:
 model: 로드된 모델
 tokenizer: 토크나이저
 user_input: 사용자 입력
 max_new_tokens: 최대 생성 토큰 수
 temperature: 생성 온도
 top_p: nucleus sampling 파라미터
 
 Returns:
 생성된 응답 텍스트
 """
 # Qwen3 모델의 chat template 사용
 messages = [{"role": "user", "content": user_input}]
 
 # Chat template 적용
 text = tokenizer.apply_chat_template(
 messages,
 tokenize=False,
 add_generation_prompt=True,
 )
 
 # 토크나이징
 model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
 
 # 생성
 with torch.no_grad():
 generated_ids = model.generate(
 **model_inputs,
 max_new_tokens=max_new_tokens,
 temperature=temperature,
 top_p=top_p,
 do_sample=True,
 pad_token_id=tokenizer.eos_token_id,
 )
 
 # 디코딩
 generated_ids = [
 output_ids[len(input_ids):] for input_ids, output_ids in zip(
 model_inputs.input_ids, generated_ids
 )
 ]
 
 response = tokenizer.batch_decode(
 generated_ids,
 skip_special_tokens=True,
 clean_up_tokenization_spaces=False,
 )[0]
 
 return response.strip()

def main():
 parser = argparse.ArgumentParser(description="Test LoRA adapter with chat")
 parser.add_argument(
 "--character",
 type=str,
 default="snow_white",
 help="Character name (default: snow_white)",
 )
 parser.add_argument(
 "--base_model",
 type=str,
 default="Qwen/Qwen3-4B-Instruct-2507",
 help="Base model path (default: Qwen/Qwen3-4B-Instruct-2507)",
 )
 parser.add_argument(
 "--adapter_path",
 type=str,
 help="Adapter path (default: checkpoints/{character})",
 )
 parser.add_argument(
 "--hf_token",
 type=str,
 help="HuggingFace token (or use HF_TOKEN env var)",
 )
 parser.add_argument(
 "--no_4bit",
 action="store_true",
 help="Disable 4-bit quantization (use full precision)",
 )
 parser.add_argument(
 "--max_new_tokens",
 type=int,
 default=512,
 help="Maximum new tokens to generate (default: 512)",
 )
 parser.add_argument(
 "--temperature",
 type=float,
 default=0.7,
 help="Generation temperature (default: 0.7)",
 )
 parser.add_argument(
 "--top_p",
 type=float,
 default=0.9,
 help="Nucleus sampling top_p (default: 0.9)",
 )
 
 args = parser.parse_args()
 
 # 어댑터 경로 설정
 if args.adapter_path is None:
 script_dir = Path(__file__).parent
 project_root = script_dir.parent
 args.adapter_path = str(project_root / "checkpoints" / args.character)
 
 logger.info("=" * 60)
 logger.info("LoRA Adapter Chat Test")
 logger.info("=" * 60)
 logger.info(f"Character: {args.character}")
 logger.info(f"Base model: {args.base_model}")
 logger.info(f"Adapter path: {args.adapter_path}")
 logger.info("=" * 60)
 
 # 모델 및 어댑터 로드
 model, tokenizer = load_model_with_adapter(
 base_model_path=args.base_model,
 adapter_path=args.adapter_path,
 hf_token=args.hf_token,
 use_4bit=not args.no_4bit,
 )
 
 # 테스트 대화
 test_prompts = [
 "안녕하세요!",
 "당신은 누구인가요?",
 "좋아하는 음식이 뭐예요?",
 "오늘 기분이 어때요?",
 "자기소개를 해주세요.",
 ]
 
 logger.info("\n" + "=" * 60)
 logger.info("Starting chat test...")
 logger.info("=" * 60 + "\n")
 
 for i, prompt in enumerate(test_prompts, 1):
 logger.info(f"[Test {i}/{len(test_prompts)}]")
 logger.info(f" User: {prompt}")
 
 try:
 response = chat(
 model=model,
 tokenizer=tokenizer,
 user_input=prompt,
 max_new_tokens=args.max_new_tokens,
 temperature=args.temperature,
 top_p=args.top_p,
 )
 logger.info(f"Assistant: {response}\n")
 except Exception as e:
 logger.error(f"Error during generation: {e}\n")
 
 logger.info("=" * 60)
 logger.info("Chat test completed!")
 logger.info("=" * 60)

if __name__ == "__main__":
 main()

