import atexit
import re
import requests
import aiohttp
from typing import AsyncIterator, List, Dict, Any
from loguru import logger
from .openai_compatible_llm import AsyncLLM


class OllamaLLM(AsyncLLM):
    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
        keep_alive: float = -1,
        unload_at_exit: bool = True,
        max_tokens: int | None = None,
    ):
        self.keep_alive = keep_alive
        self.unload_at_exit = unload_at_exit
        self.cleaned = False
        super().__init__(
            model=model,
            base_url=base_url,
            llm_api_key=llm_api_key,
            organization_id=organization_id,
            project_id=project_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            # preload model
            logger.info("Preloading model for Ollama")
            # Send the POST request to preload model
            logger.debug(
                requests.post(
                    base_url.replace("/v1", "") + "/api/chat",
                    json={
                        "model": model,
                        "keep_alive": keep_alive,
                    },
                )
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to preload model: {e}")
            logger.critical(
                "Fail to connect to Ollama backend. Is Ollama server running? Try running `ollama list` to start the server and try again.\nThe AI will repeat 'Error connecting chat endpoint' until the server is running."
            )
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
        # If keep_alive is less than 0, register cleanup to unload the model
        if unload_at_exit:
            atexit.register(self.cleanup)

    def __del__(self):
        """Destructor to unload the model"""
        self.cleanup()

    def cleanup(self):
        """Clean up function to unload the model when exitting"""
        if not self.cleaned and self.unload_at_exit:
            logger.info(f"Ollama: Unloading model: {self.model}")
            # Unload the model
            # unloading is just the same as preload, but with keep alive set to 0
            logger.debug(
                requests.post(
                    self.base_url.replace("/v1", "") + "/api/chat",
                    json={
                        "model": self.model,
                        "keep_alive": 0,
                    },
                )
            )
            self.cleaned = True

    def _fix_spacing(self, text: str) -> str:
        """
        Fix spacing after punctuation marks and honorific endings in Korean text.
        """
        if not isinstance(text, str) or not text:
            return text
        
        fixed_text = text
        honorific_endings = ["어요", "아요", "해요", "예요", "세요", "습니다", "네요", "죠", "까요", "나요", "가요", "지요", "었어요", "았어요", "했습니다"]
        
        for _ in range(50):
            # 쉼표 뒤에 공백이 없는 경우
            new_text = re.sub(r'([,])([^\s\n])', r'\1 \2', fixed_text)
            # 구두점(마침표, 물음표, 느낌표) 뒤에 공백이 없는 경우
            new_text = re.sub(r'([.!?])([^\s\n])', r'\1 \2', new_text)
            # 연속된 구두점 처리
            new_text = re.sub(r'([.,!?]{2,})([^\s\n])', r'\1 \2', new_text)
            # 한글 뒤에 구두점이 붙어있고 그 뒤에 공백이 없는 경우
            new_text = re.sub(r'([가-힣])([.,!?])([^\s\n])', r'\1\2 \3', new_text)
            # 존댓말 어미 뒤에 한글이 바로 오는 경우
            for ending in honorific_endings:
                pattern = re.escape(ending) + r'([가-힣])'
                replacement = ending + r' \1'
                new_text = re.sub(pattern, replacement, new_text)
            # 숫자 뒤의 마침표는 제외
            new_text = re.sub(r'(\d)\. (\d)', r'\1.\2', new_text)
            
            if new_text == fixed_text:
                break
            fixed_text = new_text
        return fixed_text

    def _remove_duplicates(self, text: str) -> str:
        """
        Remove duplicate text patterns from accumulated text.
        여러 패턴을 체크하여 다양한 반복을 감지하고 제거.
        """
        if not isinstance(text, str) or len(text) < 10:
            return text
        
        original_text = text
        text_len = len(text)
        
        # Check 1: exact full duplication (first half == second half)
        # 여러 split ratio 체크: 0.4, 0.45, 0.5, 0.55, 0.6
        for ratio in [0.4, 0.45, 0.5, 0.55, 0.6]:
            split_point = int(text_len * ratio)
            if split_point >= 10:
                first_part = text[:split_point]
                second_part = text[split_point:]
                # 정확히 반복되는지 체크
                if second_part.startswith(first_part):
                    remaining = second_part[len(first_part):].strip()
                    # 남은 부분이 거의 없거나 매우 짧으면 반복으로 간주
                    if len(remaining) < len(first_part) * 0.1:
                        logger.debug(f"Removed exact duplicate text (ratio {ratio:.2f})")
                        return first_part.strip()
        
        # Check 2: 전체 텍스트 반복 감지 (ABCABC 패턴)
        # 다양한 길이의 구문이 반복되는지 체크
        for phrase_len in range(50, 9, -5):  # 50자부터 10자까지 5자씩 감소
            if text_len < phrase_len * 2:
                continue
            # 마지막 phrase_len 문자와 그 이전의 동일한 길이 구문 비교
            last_phrase = text[-phrase_len:]
            # 첫 번째 절반에서 동일한 구문 찾기
            search_area = text[:text_len // 2]
            if last_phrase in search_area:
                first_idx = search_area.find(last_phrase)
                if first_idx >= 0:
                    # 반복이 시작되는 지점 찾기
                    # first_idx부터 시작하여 반복이 얼마나 긴지 확인
                    repeat_start = first_idx
                    repeat_end = first_idx + phrase_len
                    # 반복이 끝나는 지점 확인
                    if repeat_end <= text_len // 2:
                        # 반복 제거: 첫 번째 반복만 남기고 나머지 제거
                        result = text[:repeat_end].strip()
                        logger.debug(f"Removed duplicate phrase (length {phrase_len})")
                        return result
        
        # Check 3: 문장 단위 반복 감지
        # 마지막 1-3문장이 이전 문장들과 반복되는지 체크
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) >= 4:  # 최소 4문장 이상일 때만 체크
            # 마지막 3문장 체크
            for check_count in [3, 2, 1]:
                if len(sentences) < check_count * 2:
                    continue
                last_sentences = ' '.join(sentences[-check_count:])
                # 이전 문장들에서 동일한 패턴 찾기
                for i in range(len(sentences) - check_count * 2):
                    prev_sentences = ' '.join(sentences[i:i+check_count])
                    if prev_sentences == last_sentences and len(last_sentences) >= 10:
                        # 반복 제거: 첫 번째 반복만 남기고 나머지 제거
                        result = ' '.join(sentences[:i+check_count]).strip()
                        logger.debug(f"Removed duplicate sentences (count {check_count})")
                        return result
        
        # Check 4: 끝부분이 시작 부분과 반복되는지 체크
        # 마지막 30%가 첫 50%에 나타나는지 체크
        if text_len >= 20:
            check_len = min(text_len // 3, 50)  # 마지막 30% 또는 최대 50자
            last_part = text[-check_len:]
            search_text = text[:text_len // 2]  # 첫 절반에서만 검색
            
            if len(last_part) >= 10 and last_part in search_text:
                first_idx = search_text.find(last_part)
                # 시작 부분 근처에 나타나면 반복으로 간주
                if first_idx >= 0 and first_idx <= len(search_text) * 0.3:
                    result = text[:first_idx + len(last_part)].strip()
                    logger.debug(f"Removed duplicate phrase at end: '{last_part[:30]}...'")
                    return result
        
        # If text was not modified, return original
        return text

    def _ensure_sentence_completion(self, text: str) -> str:
        """
        Ensure the text ends with a complete sentence.
        If the last sentence is incomplete (doesn't end with punctuation), remove it.
        This prevents text from being cut off mid-sentence due to token limits.
        """
        if not isinstance(text, str) or len(text) < 5:
            return text
        
        text = text.strip()
        if not text:
            return text
        
        # Check if text ends with punctuation (complete sentence)
        if text[-1] in ['.', '!', '?', '。', '！', '？']:
            return text
        
        # Find the last complete sentence (ends with punctuation followed by space or end)
        # Use regex to find sentence boundaries
        # Pattern: 구두점 뒤 공백 또는 텍스트 끝
        sentence_pattern = r'[.!?。！？][\s]*'
        matches = list(re.finditer(sentence_pattern, text))
        
        if matches:
            # Get the last match (last complete sentence)
            last_match = matches[-1]
            # Include the punctuation and everything before it
            result = text[:last_match.end()].strip()
            
            # 마지막 문장이 완성되지 않았으면 무조건 제거 (더 엄격하게)
            # 불완전한 문장은 출력하지 않는 것이 좋음
            removed_length = len(text) - len(result)
            if removed_length > 0:  # 불완전한 문장이 있으면 무조건 제거
                logger.debug(f"Removed incomplete sentence ({removed_length} chars). Original: '{text[:60]}...', Result: '{result[:60]}...'")
                return result
        
        # If no sentence boundaries found, check if text is very short (might be acceptable)
        if len(text) < 15:
            return text
        
        # Try to find any punctuation mark and cut there
        last_punct_idx = max(
            text.rfind('.'),
            text.rfind('!'),
            text.rfind('?'),
            text.rfind('。'),
            text.rfind('！'),
            text.rfind('？')
        )
        
        if last_punct_idx > 0:
            # Keep text up to and including the last punctuation
            result = text[:last_punct_idx + 1].strip()
            removed_length = len(text) - len(result)
            # 불완전한 문장이 있으면 무조건 제거 (더 엄격하게)
            if removed_length > 0:
                logger.debug(f"Removed incomplete sentence after last punctuation ({removed_length} chars). Original: '{text[:60]}...', Result: '{result[:60]}...'")
                return result
        
        # If we can't find a good cut point, return original
        # (better to show incomplete text than nothing, but log it)
        if len(text) > 30:  # 긴 텍스트인데 구두점이 없으면 경고
            logger.debug(f"Text doesn't end with punctuation but keeping it: '{text[:60]}...'")
        return text

    def _extract_character_name(self, persona_prompt: str) -> str:
        """
        Extract character name from persona_prompt.
        Pattern: "당신은 ... {name}입니다" or "당신은 {name}입니다"
        """
        # Try to extract from common patterns
        patterns = [
            r"당신은\s+([가-힣]{2,10})\s*입니다",
            r"당신은\s+([가-힣]{2,10})\s*입니다",
            r"주인공\s+([가-힣]{2,10})",
            r"캐릭터\s+([가-힣]{2,10})",
        ]
        for pattern in patterns:
            match = re.search(pattern, persona_prompt)
            if match:
                return match.group(1)
        # Default fallback
        return "백설공주"

    def _extract_user_name(self, messages: List[Dict[str, Any]]) -> str | None:
        """
        Extract user name from messages.
        Patterns: "나는 {name}이야", "나는 {name}입니다", "제 이름은 {name}입니다" 등
        """
        name_patterns = [
            r"나는\s+([가-힣]{2,4})이야",
            r"나는\s+([가-힣]{2,4})입니다",
            r"제\s+이름은\s+([가-힣]{2,4})입니다",
            r"내\s+이름은\s+([가-힣]{2,4})",
            r"저는\s+([가-힣]{2,4})입니다",
            r"저는\s+([가-힣]{2,4})이에요",
        ]
        
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                
                # Handle OpenAI multimodal message format (content can be a list)
                if isinstance(content, list):
                    # Extract text from list of content items
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")
                    content = text_content
                
                # Ensure content is a string
                if not isinstance(content, str):
                    continue
                
                for pattern in name_patterns:
                    match = re.search(pattern, content)
                    if match:
                        name = match.group(1)
                        logger.debug(f"Extracted user name: {name}")
                        return name
        return None

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Override chat_completion to use /api/generate endpoint with trainable-agents format.
        This matches the training data format exactly.
        """
        # Extract persona_prompt from system
        persona_prompt = system or ""
        
        # Log persona_prompt to verify it's being passed correctly
        logger.debug(f"Persona prompt received (first 200 chars): {persona_prompt[:200]}...")
        logger.debug(f"Persona prompt length: {len(persona_prompt)}")
        
        # Extract character name from persona_prompt
        character_name = self._extract_character_name(persona_prompt)
        logger.debug(f"Extracted character name: {character_name}")
        
        # Extract user name from messages and add to persona_prompt
        user_name = self._extract_user_name(messages)
        if user_name:
            name_instruction = f"\n\n대화하고 있는 사람의 이름은 '{user_name}'입니다. 이 사람을 부를 때는 '{user_name}님'이라고 자연스럽게 부르세요. 이름 뒤에 '이'를 붙이지 말고 '님'을 붙여서 존댓말로 부르세요."
            persona_prompt = persona_prompt + name_instruction
            logger.debug(f"Added user name instruction to persona_prompt: {user_name}님")

        # Convert messages to trainable-agents format
        # Format: "{persona_prompt}\n\n사용자 말하기: {msg1}<|eot|>\n{character_name} 말하기: {response1}<|eot|>\n사용자 말하기: {msg2}<|eot|>\n{character_name} 말하기:"
        # This matches the training data format exactly
        prompt_parts = [persona_prompt]
        
        logger.debug(f"Processing {len(messages)} messages for prompt construction")
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Handle OpenAI multimodal message format (content can be a list)
            if isinstance(content, list):
                # Extract text from list of content items
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                content = text_content
            
            # Ensure content is a string
            if not isinstance(content, str):
                content = ""
            
            # Skip empty content
            if not content.strip():
                logger.debug(f"Skipping empty {role} message")
                continue
            
            if role == "user":
                prompt_parts.append(f"사용자 말하기: {content}<|eot|>")
                logger.debug(f"Added user message: {content[:50]}...")
            elif role == "assistant":
                prompt_parts.append(f"{character_name} 말하기: {content}<|eot|>")
                logger.debug(f"Added assistant message: {content[:50]}...")
            else:
                logger.debug(f"Skipping message with role: {role}")
        
        # Add the final prompt for the model to continue
        prompt_parts.append(f"{character_name} 말하기:")
        
        full_prompt = "\n".join(prompt_parts)
        logger.debug(f"Full prompt (first 500 chars): {full_prompt[:500]}")
        logger.debug(f"Total prompt parts: {len(prompt_parts)}")

        # Prepare request data for /api/generate
        request_data = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "temperature": self.temperature,
            "options": {
                "num_predict": self.max_tokens if self.max_tokens is not None else 200,  # 토큰 수 증가 (150 -> 200, trainable-agents는 300 사용)
                "repeat_penalty": 1.1,  # 반복 방지 (trainable-agents와 동일)
                "stop": [
                    "<|eot|>",
                    "<|eot_id|>",
                    "\n사용자:",
                    "\n사용자 말하기:",
                    f"\n{character_name}:",
                    f"\n{character_name} 말하기:",
                ],
            },
        }

        # Use /api/generate endpoint
        generate_url = self.base_url.replace("/v1", "") + "/api/generate"
        logger.debug(f"Calling Ollama /api/generate: {generate_url}")

        try:
            import json
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    generate_url,
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    response.raise_for_status()
                    
                    accumulated_text = ""
                    last_yielded_text = ""
                    buffer = ""
                    stopped = False
                    # 실시간 반복 감지를 위한 변수
                    seen_phrases = set()  # 이미 본 구문 추적
                    phrase_buffer = ""  # 최근 구문 버퍼 (반복 감지용)
                    # Korean honorific endings for sentence completion check (최종 처리에서만 사용)
                    HONORIFIC_ENDINGS = ["어요", "아요", "해요", "예요", "세요", "습니다", "네요", "죠", "까요", "나요", "가요", "지요", "었어요", "았어요", "했습니다"]
                    # Early stopping 비활성화: 프롬프트로만 제어하도록 변경
                    # 코드로 강제로 줄이면 대화가 망가질 수 있으므로, 프롬프트와 max_tokens만으로 제어
                    
                    async for chunk_bytes in response.content.iter_any():
                        if not chunk_bytes:
                            continue
                        
                        # Decode chunk and add to buffer
                        buffer += chunk_bytes.decode("utf-8", errors="ignore")
                        
                        # Process complete lines (JSON objects are separated by newlines)
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            
                            if not line:
                                continue
                            
                            try:
                                # Parse JSON response (each line is a JSON object)
                                chunk_data = json.loads(line)
                                
                                # Extract response text
                                if "response" in chunk_data:
                                    chunk_text = chunk_data["response"]
                                    
                                    if chunk_text and not stopped:
                                        # 실시간 반복 감지: accumulated_text에 추가하기 전에 반복 체크
                                        test_text = accumulated_text + chunk_text
                                        
                                        # 전체 텍스트 반복 체크 (정확히 절반 반복)
                                        if len(test_text) >= 20:
                                            half_len = len(test_text) // 2
                                            first_half = test_text[:half_len]
                                            second_half = test_text[half_len:]
                                            if second_half.startswith(first_half) and len(first_half) >= 10:
                                                logger.warning(f"Detected full text repetition in streaming, stopping early")
                                                stopped = True
                                                # 첫 번째 절반만 사용
                                                accumulated_text = first_half
                                                # 이미 yield한 부분 이후의 새로운 부분만 yield
                                                if accumulated_text and accumulated_text != last_yielded_text:
                                                    if last_yielded_text and accumulated_text.startswith(last_yielded_text):
                                                        new_part = accumulated_text[len(last_yielded_text):].strip()
                                                        if new_part:
                                                            yield new_part
                                                            last_yielded_text = accumulated_text
                                                    elif not last_yielded_text:
                                                        yield accumulated_text
                                                        last_yielded_text = accumulated_text
                                                break
                                        
                                        # 반복이 감지되지 않았으면 정상적으로 처리
                                        if not stopped:
                                            accumulated_text += chunk_text
                                            
                                            # Check for stop sequences in accumulated text
                                            # Stop sequence는 모델이 대화 구조를 생성하는 것을 방지하기 위해 필요
                                            should_stop = False
                                            stop_sequences = [
                                                "<|eot|>",
                                                "<|eot_id|>",
                                                "\n사용자:",
                                                "\n사용자 말하기:",
                                                f"\n{character_name}:",
                                                f"\n{character_name} 말하기:",
                                            ]
                                            # 부분적으로 포함된 stop sequence도 체크 (예: "<|eot" 같은 경우)
                                            if "<|eot" in accumulated_text and "<|eot|>" not in accumulated_text:
                                                logger.debug("Partial stop sequence '<|eot' detected, stopping generation")
                                                accumulated_text = accumulated_text.split("<|eot")[0]
                                                should_stop = True
                                                stopped = True
                                            else:
                                                for stop_seq in stop_sequences:
                                                    if stop_seq in accumulated_text:
                                                        logger.debug(f"Stop sequence '{stop_seq}' detected, stopping generation")
                                                        # Remove stop sequence and everything after it
                                                        # 주의: stop sequence 제거를 위해 accumulated_text 수정은 필요함 (예외적 경우)
                                                        accumulated_text = accumulated_text.split(stop_seq)[0]
                                                        should_stop = True
                                                        stopped = True
                                                        break
                                            
                                            if should_stop:
                                                # Yield remaining content before stop sequence (only if not already yielded)
                                                # stop sequence 제거 후 띄어쓰기 수정도 적용
                                                if accumulated_text and accumulated_text != last_yielded_text:
                                                    # stop sequence 제거된 텍스트에 띄어쓰기 수정 적용
                                                    cleaned_text = self._fix_spacing(accumulated_text)
                                                    # 추가로 구두점 뒤 띄어쓰기 확인
                                                    cleaned_text = re.sub(r'([.!?])([가-힣])', r'\1 \2', cleaned_text)
                                                    cleaned_text = re.sub(r'([,])([가-힣])', r'\1 \2', cleaned_text)
                                                    
                                                    if last_yielded_text and cleaned_text.startswith(last_yielded_text):
                                                        new_part = cleaned_text[len(last_yielded_text):].strip()
                                                        if new_part:
                                                            yield new_part
                                                            last_yielded_text = cleaned_text
                                                    elif not last_yielded_text:
                                                        yield cleaned_text
                                                        last_yielded_text = cleaned_text
                                                break
                                            
                                            # Yield chunk only if we haven't stopped
                                            # IMPORTANT: 스트리밍 중에는 원본 chunk_text를 yield하되, 실시간 반복 감지는 수행
                                            if not stopped:
                                                # Yield the original chunk_text
                                                yield chunk_text
                                                # last_yielded_text는 실제로 yield한 텍스트만 추적
                                                if not last_yielded_text:
                                                    last_yielded_text = chunk_text
                                                else:
                                                    last_yielded_text += chunk_text
                                
                                # Check if done
                                if chunk_data.get("done", False):
                                    # 최종 처리: 반복 제거 및 띄어쓰기 수정
                                    if accumulated_text and not stopped:
                                        # Remove any stop sequences from final text (모든 stop sequence를 확실히 제거)
                                        final_text = accumulated_text
                                        # 모든 stop sequence를 체크하고 제거 (break 없이 모든 것을 제거)
                                        stop_sequences = [
                                            "<|eot|>",
                                            "<|eot_id|>",
                                            "\n사용자:",
                                            "\n사용자 말하기:",
                                            f"\n{character_name}:",
                                            f"\n{character_name} 말하기:",
                                        ]
                                        # 각 stop sequence를 순차적으로 제거
                                        for stop_seq in stop_sequences:
                                            if stop_seq in final_text:
                                                # stop sequence와 그 이후의 모든 것을 제거
                                                final_text = final_text.split(stop_seq)[0]
                                                logger.debug(f"Removed stop sequence '{stop_seq}' from final text")
                                        # 추가로 부분적으로 포함된 stop sequence도 제거 (예: "<|eot" 같은 경우)
                                        if "<|eot" in final_text:
                                            final_text = final_text.split("<|eot")[0]
                                            logger.debug("Removed partial stop sequence '<|eot' from final text")
                                        # 단일 문자 stop sequence도 제거 (예: "<" 만 있는 경우)
                                        if final_text.endswith("<"):
                                            final_text = final_text[:-1].strip()
                                            logger.debug("Removed single character stop sequence '<' from final text")
                                        final_text = final_text.strip()
                                        
                                        # 문장 완성 체크: 마지막 문장이 완성되지 않았으면 제거
                                        # 토큰 제한으로 인해 문장이 중간에 끊긴 경우를 방지
                                        final_text = self._ensure_sentence_completion(final_text)
                                        
                                        # 반복 제거 수행 (여러 번 반복하여 중첩 반복까지 제거)
                                        text_stripped = final_text
                                        for _ in range(3):  # 최대 3번 반복하여 중첩 반복까지 제거
                                            new_text = self._remove_duplicates(text_stripped)
                                            if new_text == text_stripped:
                                                break
                                            text_stripped = new_text
                                        
                                        # 띄어쓰기 수정 (강력하게 적용)
                                        text_stripped = self._fix_spacing(text_stripped)
                                        # 추가로 구두점 뒤 띄어쓰기 확인 및 수정
                                        text_stripped = re.sub(r'([.!?])([가-힣])', r'\1 \2', text_stripped)
                                        text_stripped = re.sub(r'([,])([가-힣])', r'\1 \2', text_stripped)
                                        text_stripped = text_stripped.strip()
                                        
                                        # 띄어쓰기 수정 후 다시 stop sequence 체크 (띄어쓰기 수정으로 인해 노출될 수 있음)
                                        if "<|eot" in text_stripped:
                                            text_stripped = text_stripped.split("<|eot")[0].strip()
                                        if text_stripped.endswith("<"):
                                            text_stripped = text_stripped[:-1].strip()
                                        
                                        # 이미 yield한 부분과 비교하여 새로운 부분만 yield
                                        # 중요: last_yielded_text는 이미 yield한 전체 텍스트이므로,
                                        # text_stripped가 last_yielded_text로 시작하면 새로운 부분만 yield
                                        if text_stripped and text_stripped != last_yielded_text:
                                            if last_yielded_text and text_stripped.startswith(last_yielded_text):
                                                # 새로운 부분만 yield
                                                new_part = text_stripped[len(last_yielded_text):].strip()
                                                if new_part:
                                                    logger.debug(f"Yielding final new part: '{new_part[:50]}...'")
                                                    yield new_part
                                            elif not last_yielded_text:
                                                # 아직 아무것도 yield하지 않았으면 전체 yield
                                                logger.debug(f"Yielding final text (no previous yield): '{text_stripped[:50]}...'")
                                                yield text_stripped
                                            else:
                                                # 완전히 다른 텍스트인 경우 (이상한 경우지만 안전장치)
                                                logger.warning(f"Final text doesn't start with last_yielded_text. Last: '{last_yielded_text[:50]}...', Final: '{text_stripped[:50]}...'")
                                                # 새로운 부분만 yield하려고 시도
                                                # 공통 접두사 찾기
                                                common_prefix_len = 0
                                                min_len = min(len(last_yielded_text), len(text_stripped))
                                                for i in range(min_len):
                                                    if last_yielded_text[i] == text_stripped[i]:
                                                        common_prefix_len += 1
                                                    else:
                                                        break
                                                if common_prefix_len > 0:
                                                    new_part = text_stripped[common_prefix_len:].strip()
                                                    if new_part:
                                                        yield new_part
                                    break
                                    
                            except json.JSONDecodeError as e:
                                logger.debug(f"Failed to parse JSON line: {line[:100]}, error: {e}")
                                continue
                            except Exception as e:
                                logger.debug(f"Error processing line: {e}")
                                continue

        except aiohttp.ClientError as e:
            logger.error(f"Error calling Ollama /api/generate: {e}")
            yield f"Error calling the chat endpoint: {e}"
        except Exception as e:
            logger.error(f"Unexpected error in chat_completion: {e}")
            yield f"Error calling the chat endpoint: {e}"
