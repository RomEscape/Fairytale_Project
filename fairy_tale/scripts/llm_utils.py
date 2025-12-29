"""
Ollama LLM 호출 유틸리티

장면 및 대화 데이터 생성에 사용할 LLM 호출 함수들을 제공합니다.
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional
from loguru import logger


class OllamaLLMClient:
    """Ollama LLM 클라이언트"""
    
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        """
        초기화
        
        Args:
            model: 사용할 Ollama 모델명
            base_url: Ollama 서버 URL
            temperature: 생성 온도
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.chat_url = f"{self.base_url}/api/chat"
        
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        language: str = "korean",
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            system: 시스템 프롬프트 (선택)
            max_tokens: 최대 토큰 수 (선택)
            temperature: 생성 온도 (선택, None이면 기본값 사용)
            language: 응답 언어 (기본값: korean)
            
        Returns:
            생성된 텍스트
        """
        messages = []
        
        # 시스템 프롬프트 설정 (언어 지시 포함)
        if system:
            if language == "korean":
                system_with_lang = f"{system}\n\n중요: 반드시 한국어로만 응답하세요. 모든 출력은 한국어여야 합니다."
            else:
                system_with_lang = system
            messages.append({"role": "system", "content": system_with_lang})
        elif language == "korean":
            # 시스템 프롬프트가 없는 경우에도 언어 지시 추가
            messages.append({
                "role": "system",
                "content": "당신은 한국어로 응답하는 도우미 AI입니다. 모든 응답은 반드시 한국어로 작성해야 합니다."
            })
        
        # 프롬프트에 언어 지시 추가
        if language == "korean" and "한국어" not in prompt and "korean" not in prompt.lower():
            prompt = f"{prompt}\n\n중요: 반드시 한국어로만 응답하세요. 모든 장면 설명은 한국어로 작성해야 합니다."
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        
        # max_tokens 기본값 증가 (장면 생성은 긴 응답 필요)
        default_max_tokens = max_tokens if max_tokens else 8192
        
        payload["options"] = {
            "num_predict": default_max_tokens,
            "num_ctx": 16384,  # 컨텍스트 윈도우 증가
        }
        
        try:
            logger.debug(f"Calling Ollama: model={self.model}, prompt_length={len(prompt)}, max_tokens={default_max_tokens}")
            # 타임아웃 증가: 긴 응답 생성을 위해
            response = requests.post(self.chat_url, json=payload, timeout=600)
            response.raise_for_status()
            
            result = response.json()
            
            # 디버깅: 응답 구조 전체 로깅 (빈 응답 문제 진단용)
            if "message" not in result or "content" not in result.get("message", {}):
                logger.error(f"Unexpected response format: {json.dumps(result, ensure_ascii=False, indent=2)}")
                # 에러 메시지가 있는지 확인
                if "error" in result:
                    logger.error(f"Ollama API error: {result['error']}")
                raise ValueError(f"Unexpected response format: {result}")
            
            generated_text = result["message"]["content"]
            logger.debug(f"Generated text length: {len(generated_text)}")
            
            # 빈 응답인 경우 더 자세한 정보 로깅
            if not generated_text or len(generated_text.strip()) == 0:
                logger.warning(
                    f"Empty response from Ollama. Full response: {json.dumps(result, ensure_ascii=False)[:500]}"
                )
                # done 필드 확인 (스트리밍이 끝났는지)
                if result.get("done", False):
                    logger.warning("Response marked as 'done' but content is empty - model may have failed to generate")
                return ""  # 빈 문자열 반환하여 상위에서 처리하도록
            
            # 응답이 불완전한지 검사 (너무 짧거나 중간에 끊긴 경우)
            if len(generated_text.strip()) < 100:
                logger.warning(
                    f"Generated text is very short ({len(generated_text)} chars), might be incomplete. "
                    f"Content preview: {generated_text[:200]}"
                )
            
            # 한국어 응답 검증 (언어가 korean인 경우)
            if language == "korean":
                korean_char_count = sum(1 for c in generated_text if '\uAC00' <= c <= '\uD7A3')
                total_char_count = len([c for c in generated_text if c.isalnum() or c in [' ', '\n', '\t']])
                if total_char_count > 0:
                    korean_ratio = korean_char_count / total_char_count
                    if korean_ratio < 0.3:
                        logger.warning(
                            f"Generated text might not be in Korean (Korean char ratio: {korean_ratio:.2%}). "
                            f"Content preview: {generated_text[:200]}"
                        )
            
            return generated_text.strip()
                
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Failed to connect to Ollama at {self.base_url}. "
                "Is Ollama server running? Run 'ollama serve' to start."
            )
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LLM generation: {e}")
            raise
    
    def check_server(self) -> bool:
        """
        Ollama 서버 연결 확인
        
        Returns:
            연결 성공 여부
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


def load_prompt_template(prompt_file: str, **kwargs) -> str:
    """
    프롬프트 템플릿 파일 로드 및 변수 치환
    
    Args:
        prompt_file: 프롬프트 파일 경로
        **kwargs: 템플릿 변수 값들
        
    Returns:
        변수 치환이 완료된 프롬프트 문자열
    """
    from pathlib import Path
    
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # 변수 치환
    try:
        formatted_prompt = template.format(**kwargs)
    except KeyError as e:
        logger.warning(f"Template variable {e} not provided, using as-is")
        formatted_prompt = template
    
    return formatted_prompt


def load_profile(profile_file: str) -> tuple[str, list[str]]:
    """
    프로파일 파일 로드 (trainable-agents 방식)
    
    파일 구조:
    - 1번 줄: 캐릭터 이름
    - 2번 줄: 빈 줄
    - 3번 줄부터: 프로필 문단들 (유효한 모든 문단)
    
    Args:
        profile_file: 프로파일 파일 경로
        
    Returns:
        (캐릭터 이름, 문단 리스트) - 3번 줄부터 끝까지의 모든 유효한 문단 반환
    """
    from pathlib import Path
    
    profile_path = Path(profile_file)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_file}")
    
    with open(profile_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    # 빈 줄로 문단 분리
    parts = content.split("\n\n")
    
    # 첫 부분에서 캐릭터 이름 추출 (1번 줄)
    first_part = parts[0].strip()
    if first_part.startswith("# "):
        character_name = first_part.replace("#", "").strip()
    else:
        character_name = first_part.strip()
    
    # 3번 줄부터 시작하는 프로필 문단 추출 (유효한 모든 문단)
    # parts[1:]은 2번 줄 이후 빈 줄로 분리된 모든 문단
    # 1-2번 줄을 제외하고 3번 줄부터 끝까지의 모든 유효한 문단 사용
    agent_profile = [p.strip() for p in parts[1:] if p.strip()]
    
    if not agent_profile:
        # 문단이 없는 경우
        logger.error(f"No profile paragraphs found in {profile_file}")
        # 파일명에서 캐릭터 이름 추출 시도
        character_name = profile_path.stem.replace("wiki_", "").replace("-korean", "")
        # 빈 줄이 하나도 없는 경우, 줄바꿈으로 나눔
        lines = content.split("\n")
        if len(lines) > 1:
            character_name = lines[0].strip()
            agent_profile = ["\n".join(lines[1:]).strip()]
        else:
            agent_profile = []
    
    logger.info(
        f"Loaded profile: character='{character_name}', paragraphs={len(agent_profile)} "
        f"(expecting {len(agent_profile)} × 20 = {len(agent_profile) * 20} scenes)"
    )
    
    return character_name, agent_profile

