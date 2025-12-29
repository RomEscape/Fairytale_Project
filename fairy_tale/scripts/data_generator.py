"""
LLM을 사용한 데이터 생성 모듈

Ollama를 사용하여 장면 및 대화 데이터를 생성합니다.
"""

import json
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from tqdm import tqdm

from llm_utils import OllamaLLMClient, load_prompt_template, load_profile


class SceneDataGenerator:
    """장면 데이터 생성기"""
    
    def __init__(
        self,
        character: str,
        model: str = "exaone3.5:2.4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        """
        초기화
        
        Args:
            character: 캐릭터 이름
            model: Ollama 모델명
            base_url: Ollama 서버 URL
            temperature: 생성 온도
        """
        self.character = character
        self.llm_client = OllamaLLMClient(model=model, base_url=base_url, temperature=temperature)
        self.base_dir = Path(__file__).parent.parent
        
        # 서버 연결 확인
        if not self.llm_client.check_server():
            raise RuntimeError(
                f"Ollama server is not running at {base_url}. "
                "Please start Ollama server with 'ollama serve' or check the URL."
            )
    
    def generate_scenes(
        self,
        output_path: Path,
        num_scenes: int = 40,
    ) -> Path:
        """
        장면 데이터 생성 (각 문단마다 반복)
        
        Args:
            output_path: 출력 JSONL 파일 경로
            num_scenes: 각 문단당 생성할 장면 수 (기본 40개)
            
        Returns:
            생성된 파일 경로
        """
        # 프로파일 로드
        seed_data_dir = self.base_dir / "data" / "seed_data"
        profile_file = seed_data_dir / "profiles" / f"wiki_{self.character}-korean.txt"
        
        if not profile_file.exists():
            raise FileNotFoundError(f"Profile file not found: {profile_file}")
        
        agent_name, agent_profile = load_profile(str(profile_file))
        logger.info(f"Loaded profile for {agent_name} with {len(agent_profile)} paragraphs")
        
        # 프롬프트 템플릿 로드
        prompt_file = seed_data_dir / "prompts" / f"prompt_scene_generation_{self.character}_korean.txt"
        if not prompt_file.exists():
            # 일반 프롬프트 사용
            prompt_file = seed_data_dir / "prompts" / "prompt_agent_scene_korean.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found for scene generation")
        
        logger.info(f"Using prompt template: {prompt_file.name}")
        
        # 출력 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 기존 파일 처리: 파일이 있으면 백업하고 새로 시작 (중복 방지)
        if output_path.exists():
            backup_path = output_path.with_suffix(output_path.suffix + ".backup")
            logger.info(f"Existing file found, backing up to {backup_path.name}")
            import shutil
            shutil.copy2(output_path, backup_path)
            output_path.unlink()  # 새로 시작
        
        # 유효한 문단만 필터링
        valid_paragraphs = [p for p in agent_profile if p.strip()]
        total_paragraphs = len(valid_paragraphs)
        
        if total_paragraphs == 0:
            logger.warning("No valid paragraphs found in profile")
            return output_path
        
        # Progress bar 생성 (trainable-agents 방식)
        progress_bar = tqdm(
            total=total_paragraphs,
            desc=f"Generating scenes for {agent_name}",
            unit="paragraph",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        total_scenes = 0
        failed_count = 0
        
        # 각 문단마다 반복
        max_retries = 3  # 최대 재시도 횟수
        for para_idx, paragraph in enumerate(valid_paragraphs):
            # 문단 하나에 대한 프롬프트 생성
            prompt = load_prompt_template(
                str(prompt_file),
                agent_name=agent_name,
                agent_summary=paragraph.strip(),  # 문단 하나만 전달
            )
            
            # 재시도 로직
            success = False
            completions = None
            
            for retry in range(max_retries):
                try:
                    # 재시도 시 대기 시간 추가 (exponential backoff)
                    if retry > 0:
                        wait_time = min(2 ** retry, 30)  # 최대 30초
                        logger.info(f"Waiting {wait_time}s before retry {retry + 1}/{max_retries} for paragraph {para_idx + 1}...")
                        time.sleep(wait_time)
                    
                    # LLM 호출
                    completions = self.llm_client.generate(
                        prompt=prompt,
                        max_tokens=25000,  # 40개 장면 완전 생성에 충분한 토큰 (평균 600자/장면 × 40개)
                        language="korean",  # 언어 지시 추가
                    )
                    
                    if not completions or len(completions.strip()) == 0:
                        if retry < max_retries - 1:
                            logger.warning(
                                f"Empty response for paragraph {para_idx + 1} (paragraph length: {len(paragraph)} chars, "
                                f"prompt length: {len(prompt)} chars), retry {retry + 1}/{max_retries}"
                            )
                            continue
                        else:
                            logger.error(f"Empty response for paragraph {para_idx + 1} after {max_retries} retries")
                            failed_count += 1
                            progress_bar.set_postfix({
                                "status": "empty",
                                "failed": failed_count
                            })
                            break
                    
                    # 응답 완전성 검증: 실제로 num_scenes개 장면이 모두 생성되었는지 확인
                    import re
                    # "장면 1", "장면 2", ... "장면 {num_scenes}" 패턴 찾기
                    scene_pattern = r'장면\s*(\d+)'
                    found_scenes = set()
                    for match in re.finditer(scene_pattern, completions):
                        scene_num = int(match.group(1))
                        if 1 <= scene_num <= num_scenes:
                            found_scenes.add(scene_num)
                    
                    # 축약 표현 체크
                    has_abbreviation = bool(re.search(r'장면\s*\d+\s*[~\-~]\s*\d+', completions))
                    if has_abbreviation:
                        logger.warning(f"Abbreviation detected in paragraph {para_idx + 1} response (e.g., '장면 11~{num_scenes}')")
                    
                    # num_scenes개 장면이 모두 있는지 확인
                    expected_scenes = set(range(1, num_scenes + 1))
                    missing_scenes = expected_scenes - found_scenes
                    scene_count = len(found_scenes)
                    
                    if scene_count < num_scenes:
                        if retry < max_retries - 1:
                            logger.warning(
                                f"Incomplete response for paragraph {para_idx + 1}: "
                                f"only {scene_count}/{num_scenes} scenes found (missing: {sorted(missing_scenes)[:5] if missing_scenes else 'none'}), "
                                f"retry {retry + 1}/{max_retries}"
                            )
                            continue
                        else:
                            logger.error(
                                f"Incomplete response for paragraph {para_idx + 1} after {max_retries} retries: "
                                f"only {scene_count}/{num_scenes} scenes found (missing: {sorted(missing_scenes)[:10] if missing_scenes else 'none'})"
                            )
                            failed_count += 1
                            progress_bar.set_postfix({
                                "status": "incomplete",
                                "scenes": f"{scene_count}/{num_scenes}",
                                "failed": failed_count
                            })
                            break
                    
                    if has_abbreviation:
                        logger.warning(f"Warning: Abbreviation detected but {num_scenes} scenes found. Continuing anyway.")
                    
                    # 성공 시 로그
                    if scene_count == num_scenes:
                        logger.debug(f"Paragraph {para_idx + 1}: All {num_scenes} scenes generated successfully")
                    
                    # 한국어 응답 검증
                    korean_char_count = sum(1 for c in completions if '\uAC00' <= c <= '\uD7A3')
                    if korean_char_count < 50:
                        logger.warning(f"Response might not be in Korean for paragraph {para_idx + 1} (Korean chars: {korean_char_count})")
                        # 경고만 하고 계속 진행 (재시도하지 않음)
                    
                    # 성공: 결과 저장
                    gen_id = str(uuid.uuid4())[:8]
                    output_data = {
                        "gen_answer_id": gen_id,
                        "prompt": prompt,
                        "completions": completions,
                        "check_result": True,
                    }
                    
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    
                    total_scenes += 1
                    success = True
                    progress_bar.set_postfix({
                        "status": "success",
                        "saved": total_scenes,
                        "failed": failed_count
                    })
                    break  # 성공하면 재시도 중단
                    
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"Error for paragraph {para_idx + 1}, retry {retry + 1}/{max_retries}: {e}")
                        continue
                    else:
                        logger.error(f"Error for paragraph {para_idx + 1} after {max_retries} retries: {e}")
                        failed_count += 1
                        progress_bar.set_postfix({
                            "status": "error",
                            "failed": failed_count
                        })
                        break
            
            if not success:
                logger.error(f"Failed to generate scenes for paragraph {para_idx + 1} (index: {para_idx}) after {max_retries} retries")
            
            progress_bar.update(1)
        
        progress_bar.close()
        
        # 생성 완료 후 검증: 누락된 문단이 있는지 확인
        if total_scenes < total_paragraphs:
            logger.warning(
                f"Warning: Only {total_scenes}/{total_paragraphs} paragraphs were successfully generated. "
                f"{total_paragraphs - total_scenes} paragraphs are missing. "
                "Please check the logs and regenerate if needed."
            )
        
        logger.success(f"Saved {total_scenes} scene data entries to {output_path} (failed: {failed_count}/{total_paragraphs})")
        return output_path


class DialogueDataGenerator:
    """대화 데이터 생성기"""
    
    def __init__(
        self,
        character: str,
        model: str = "exaone3.5:2.4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
    ):
        """
        초기화
        
        Args:
            character: 캐릭터 이름
            model: Ollama 모델명
            base_url: Ollama 서버 URL
            temperature: 생성 온도
        """
        self.character = character
        self.llm_client = OllamaLLMClient(model=model, base_url=base_url, temperature=temperature)
        self.base_dir = Path(__file__).parent.parent
        
        # 서버 연결 확인
        if not self.llm_client.check_server():
            raise RuntimeError(
                f"Ollama server is not running at {base_url}. "
                "Please start Ollama server with 'ollama serve' or check the URL."
            )
    
    def load_scenes(self, scene_file_path: Path | None = None) -> List[Dict[str, Any]]:
        """
        파싱된 장면 데이터 로드
        
        Args:
            scene_file_path: 장면 파일 경로 (None이면 자동으로 찾음)
            
        Returns:
            장면 데이터 리스트
        """
        if scene_file_path is None:
            # processed 디렉토리에서 가장 최근 날짜의 scene_*.json 파일 찾기
            processed_dir = self.base_dir / "processed"
            if not processed_dir.exists():
                raise FileNotFoundError(
                    f"Processed directory not found: {processed_dir}. "
                    "Please generate and parse scenes first."
                )
            
            # 날짜별 디렉토리 찾기 (가장 최근 것)
            date_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}', d.name)], reverse=True)
            if not date_dirs:
                raise FileNotFoundError(
                    f"No date directories found in {processed_dir}. "
                    "Please generate and parse scenes first."
                )
            
            # 캐릭터 이름이 포함된 scene 파일 찾기 (trainable-agents 구조)
            scene_files = list(date_dirs[0].glob(f"generated_agent_scene_{self.character}*.json"))
            if not scene_files:
                raise FileNotFoundError(
                    f"Scene file not found in {date_dirs[0]} for character {self.character}. "
                    "Please generate and parse scenes first."
                )
            
            # 가장 최근 파일 선택
            scene_file_path = max(scene_files, key=lambda p: p.stat().st_mtime)
        
        if not scene_file_path.exists():
            raise FileNotFoundError(f"Scene file not found: {scene_file_path}")
        
        with open(scene_file_path, "r", encoding="utf-8") as f:
            scenes = json.load(f)
        
        logger.info(f"Loaded {len(scenes)} scenes from {scene_file_path}")
        return scenes
    
    def generate_dialogues(
        self,
        output_path: Path,
        scene_data_path: Path | None = None,
        dialogues_per_scene: int = 23,
        resume: bool = True,
    ) -> Path:
        """
        대화 데이터 생성 (한 장면당 여러 개의 대화 데이터 생성)
        
        Args:
            output_path: 출력 JSONL 파일 경로
            scene_data_path: 처리된 장면 데이터 파일 경로 (None이면 자동으로 찾음)
            dialogues_per_scene: 각 장면당 생성할 대화 개수 (기본 23개, 조수미 데이터 수준과 비등비등)
            resume: 기존 파일이 있으면 중단된 지점부터 이어서 생성 (기본 True)
            
        Returns:
            생성된 파일 경로
        """
        logger.info(f"Generating dialogues for character: {self.character}")
        
        # 장면 로드 (명시적 경로가 있으면 사용, 없으면 자동으로 찾음)
        scenes = self.load_scenes(scene_file_path=scene_data_path)
        
        # 프로파일 로드
        seed_data_dir = self.base_dir / "data" / "seed_data"
        profile_file = seed_data_dir / "profiles" / f"wiki_{self.character}-korean.txt"
        
        if not profile_file.exists():
            raise FileNotFoundError(f"Profile file not found: {profile_file}")
        
        agent_name, agent_profile = load_profile(str(profile_file))
        
        # 전체 프로필을 하나의 텍스트로 합치기 (대화 생성에는 전체 프로필 필요)
        agent_summary = "\n\n".join(agent_profile) if isinstance(agent_profile, list) else agent_profile
        
        # 프롬프트 로드
        prompt_template_file = seed_data_dir / "prompts" / "prompt_agent_dialogue_korean.txt"
        if not prompt_template_file.exists():
            raise FileNotFoundError(f"Dialogue prompt template not found: {prompt_template_file}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 모든 장면 처리 (한 장면당 여러 개의 대화 데이터 생성)
        total_scenes = len(scenes)
        total_dialogues = total_scenes * dialogues_per_scene
        
        if total_scenes == 0:
            logger.warning("No scenes to process")
            return output_path
        
        # 기존 파일 확인 및 resume 처리
        start_scene_idx = 0
        start_dialogue_num = 0
        existing_dialogue_count = 0
        
        logger.debug(f"Checking for existing file: {output_path} (resume={resume})")
        if resume and output_path.exists():
            logger.info(f"Existing file found: {output_path}")
            try:
                logger.debug(f"Reading existing file...")
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_lines = f.readlines()
                    existing_dialogue_count = len(existing_lines)
                
                logger.debug(f"Read {existing_dialogue_count} existing dialogues")
                
                # 완료된 장면 수와 마지막 장면의 완료된 대화 수 계산
                completed_scenes = existing_dialogue_count // dialogues_per_scene
                remaining_in_last_scene = existing_dialogue_count % dialogues_per_scene
                
                if completed_scenes < total_scenes:
                    start_scene_idx = completed_scenes
                    start_dialogue_num = remaining_in_last_scene
                    logger.info(
                        f"Resuming from scene {start_scene_idx + 1}/{total_scenes}, "
                        f"dialogue {start_dialogue_num + 1}/{dialogues_per_scene} "
                        f"(existing: {existing_dialogue_count} dialogues)"
                    )
                else:
                    logger.info(f"All dialogues already generated ({existing_dialogue_count} dialogues)")
                    return output_path
            except Exception as e:
                logger.warning(f"Failed to read existing file, starting from beginning: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                start_scene_idx = 0
                start_dialogue_num = 0
        elif resume:
            logger.info(f"Resume enabled but file not found: {output_path}. Starting from beginning.")
        elif not resume:
            logger.info(f"Resume disabled. Starting from beginning.")
        
        remaining_dialogues = total_dialogues - existing_dialogue_count
        logger.info(f"Processing {total_scenes} scenes ({dialogues_per_scene} dialogues per scene, total: {total_dialogues} dialogues)")
        logger.info(f"Remaining: {remaining_dialogues} dialogues ({total_scenes - start_scene_idx} scenes)")
        
        # Progress bar 생성 (남은 대화 수 기준)
        progress_bar = tqdm(
            total=remaining_dialogues,
            desc=f"Generating dialogues for {agent_name}",
            unit="dialogue",
            initial=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        dialogue_count = 0
        failed_count = 0
        
        # 각 장면에 대해 여러 개의 대화 생성
        for scene_idx in range(start_scene_idx, len(scenes)):
            scene = scenes[scene_idx]
            scene_type = scene.get("type", "대화")
            location = scene.get("location", "")
            background = scene.get("background", "")
            
            # 캐릭터 이름 추출 (짧은 이름)
            agent_short_name = agent_name
            if len(agent_name) > 4:
                agent_short_name = agent_name[:2]  # 백설공주 -> 백설
            
            # 장면의 profile 정보가 있으면 사용, 없으면 전체 프로필 사용
            scene_profile = scene.get("profile", agent_summary)
            
            # 한 장면당 여러 개의 대화 생성
            dialogue_start = start_dialogue_num if scene_idx == start_scene_idx else 0
            for dialogue_num in range(dialogue_start, dialogues_per_scene):
                # 대화 변형 안내 문구 생성
                if dialogues_per_scene > 1:
                    dialogue_variant_instruction = (
                        f"**중요**: 이 장면에 대해 {dialogues_per_scene}개의 서로 다른 대화를 생성 중입니다. "
                        f"현재는 {dialogue_num + 1}번째 대화입니다.\n"
                        f"- 각 대화는 서로 다른 관점, 주제, 질문을 가져야 합니다.\n"
                        f"- 사용자의 질문이 매번 달라야 하고, 캐릭터의 응답도 다양한 측면을 보여야 합니다.\n"
                        f"- {dialogue_num + 1}번째 대화이므로 이전 대화와는 완전히 다른 주제나 접근 방식으로 작성하세요.\n"
                    )
                else:
                    dialogue_variant_instruction = ""
                
                prompt = load_prompt_template(
                    str(prompt_template_file),
                    agent_name=agent_name,
                    agent_short_name=agent_short_name,
                    agent_summary=scene_profile,
                    type=scene_type,
                    location=location,
                    background=background,
                    dialogue_variant_instruction=dialogue_variant_instruction,
                )
                
                try:
                    completions = self.llm_client.generate(
                        prompt=prompt,
                        max_tokens=3000,  # 1000-1500단어 생성에 충분한 토큰 (15-20개 턴)
                    )
                    
                    if not completions or len(completions.strip()) == 0:
                        failed_count += 1
                        progress_bar.set_postfix({
                            "status": "empty",
                            "failed": failed_count
                        })
                        progress_bar.update(1)
                        continue
                    
                    # 반복 문자 패턴 검증 (예: "따따따...", "가가가...")
                    # 같은 문자가 5번 이상 연속으로 나오는 패턴 확인
                    if re.search(r'(.)\1{4,}', completions):
                        logger.warning(f"Repetitive character pattern detected in dialogue {dialogue_num + 1} for scene {scene_idx + 1}, skipping")
                        failed_count += 1
                        progress_bar.set_postfix({
                            "status": "repetitive",
                            "failed": failed_count
                        })
                        progress_bar.update(1)
                        continue
                    
                    # 의미 없는 짧은 응답 필터링 (배경만 있고 대화 턴이 없는 경우)
                    if len(completions.strip()) < 100:
                        logger.warning(f"Too short response ({len(completions)} chars) in dialogue {dialogue_num + 1} for scene {scene_idx + 1}, skipping")
                        failed_count += 1
                        progress_bar.set_postfix({
                            "status": "too_short",
                            "failed": failed_count
                        })
                        progress_bar.update(1)
                        continue
                    
                    gen_id = str(uuid.uuid4())[:8]
                    output_data = {
                        "gen_answer_id": gen_id,
                        "prompt": prompt,
                        "completions": completions,
                        "check_result": True,
                        "scene_data": {
                            "location": location,
                            "background": background,
                            "type": scene_type,
                        },
                    }
                    
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    
                    dialogue_count += 1
                    progress_bar.set_postfix({
                        "status": "success",
                        "saved": dialogue_count,
                        "failed": failed_count,
                        "scene": f"{scene_idx + 1}/{total_scenes}"
                    })
                    
                except Exception as e:
                    failed_count += 1
                    progress_bar.set_postfix({
                        "status": "error",
                        "failed": failed_count
                    })
                    logger.debug(f"Error for scene {scene_idx + 1}, dialogue {dialogue_num + 1}: {e}")
                
                progress_bar.update(1)
        
        progress_bar.close()
        total_generated = existing_dialogue_count + dialogue_count
        logger.success(
            f"Generated {dialogue_count} new dialogues and saved to {output_path} "
            f"(total: {total_generated}/{total_dialogues}, failed: {failed_count})"
        )
        return output_path

