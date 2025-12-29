"""
프롬프트 데이터를 Fine-tuning 형식으로 변환

대화 데이터를 JSONL 형식으로 변환하여 QLoRA 학습에 사용합니다.
참고 프로젝트의 convert_prompt_data.py 구조를 따릅니다.
"""

import os
import json
import re
import sys
from pathlib import Path
from loguru import logger


def clean_text(text: str) -> str:
    """
    텍스트 정리 함수
    
    - 따옴표\n\n 제거
    - 연속된 \n\n 정리
    - 영어 단어 제거 (pervasive 등)
    - 기타 이상한 패턴 제거
    
    Args:
        text: 원본 텍스트
    
    Returns:
        정리된 텍스트
    """
    if not text:
        return ""
    
    # 따옴표\n\n 패턴 제거
    text = re.sub(r'["\'"]\s*\n\s*\n', '', text)
    text = re.sub(r'\n\s*\n\s*["\']', '', text)
    text = re.sub(r'["\'"]\s*\n', '', text)
    text = re.sub(r'\n\s*["\']', '', text)
    
    # 연속된 줄바꿈 정리 (최대 2개까지)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 공백과 줄바꿈이 혼합된 패턴 정리
    text = re.sub(r'\n\s+\n', '\n\n', text)
    text = re.sub(r'\s+\n\s+', '\n', text)
    
    # 영어 단어 제거 (일반적인 영어 단어는 제외)
    # 허용되는 영어 단어: 일반 전치사, 관사 등
    allowed_words = {'the', 'and', 'or', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 
                     'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 
                     'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    
    def remove_english_words(match):
        word = match.group(0)
        # 단일 문자나 허용된 단어는 유지
        if len(word) <= 2 or word.lower() in allowed_words:
            return word
        # 3자 이상의 영어 단어는 제거
        return ''
    
    # 영어 단어 뒤에 한글 조사가 붙은 경우 먼저 처리 (예: "pervasive한" -> "")
    text = re.sub(r'\b[A-Za-z]{3,}(한|의|가|을|를|은|는|이|으로|로|에|에서|에게|께|와|과|도|만|부터|까지|만큼|처럼|같이|보다)\b', '', text)
    
    # 나머지 독립적인 영어 단어 제거
    text = re.sub(r'\b[A-Za-z]{3,}\b', remove_english_words, text)
    
    # 남은 따옴표 정리
    text = re.sub(r'["\'"]+', '', text)
    
    # "따옴표\n\n", "따옴표\n", "따옴표" 리터럴 텍스트 제거
    # LLM이 실제로 "따옴표"라는 단어를 생성한 경우 제거
    text = re.sub(r'따옴표\s*\n\s*\n', '', text)
    text = re.sub(r'따옴표\s*\n', '', text)
    text = re.sub(r'^따옴표\s+', '', text)
    text = re.sub(r'\s+따옴표\s+', ' ', text)
    
    # 공백 정리 및 혼자 남은 조사 제거 (공백으로 둘러싸인 조사)
    text = re.sub(r'  +', ' ', text)
    # "처럼", "같이", "보다" 등이 혼자 남은 경우 제거
    text = re.sub(r'\s+(처럼|같이|보다)\s+', ' ', text)
    text = text.strip()
    
    return text


def read_profile(path: str) -> tuple[str, list[str]]:
    """
    프로파일 읽기
    
    Args:
        path: 프로파일 파일 경로
    
    Returns:
        (캐릭터 이름, 프로파일 내용 리스트)
    """
    with open(path, 'r', encoding='utf-8') as fp:
        text = fp.read().strip()
    
    parts = text.split('\n\n')
    # 첫 번째 줄이 캐릭터 이름 (예: "백설공주" 또는 "# 백설공주")
    first_line = parts[0].strip()
    if first_line.startswith('# '):
        agent_name = first_line.replace('# ', '').strip()
        agent_profile = [p.strip() for p in parts[1:]]
    else:
        # "# " 없이 시작하는 경우 (예: "백설공주")
        agent_name = first_line
        agent_profile = [p.strip() for p in parts[1:]]
    
    return agent_name, agent_profile


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        logger.error("Usage: python convert_prompt_data.py <dialogue_data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    # 경로 해결 (parse_data_dialogue.py와 동일한 방식)
    base_dir = Path(__file__).parent.parent
    current_dir = Path.cwd()
    project_root = base_dir.parent
    
    # 절대 경로인 경우 그대로 사용
    data_path_obj = None
    if Path(data_path).is_absolute():
        data_path_obj = Path(data_path)
    else:
        # 가능한 경로들 시도
        possible_paths = [
            Path(data_path),  # 현재 디렉토리 기준
            current_dir / data_path,  # 현재 작업 디렉토리 기준
            base_dir / data_path,  # fairy_tale/ 기준
            project_root / data_path,  # 프로젝트 루트 기준
        ]
        
        # 존재하는 첫 번째 경로 사용
        for path in possible_paths:
            if path.exists():
                data_path_obj = path
                break
    
    # 최종 경로 확인
    if data_path_obj is None or not data_path_obj.exists():
        logger.error(f"Data file not found: {sys.argv[1]}")
        logger.info(f"Current working directory: {current_dir}")
        logger.info(f"Tried paths:")
        logger.info(f"  - {Path(data_path)}")
        logger.info(f"  - {current_dir / data_path}")
        logger.info(f"  - {base_dir / data_path}")
        logger.info(f"  - {project_root / data_path}")
        sys.exit(1)
    
    data_path = str(data_path_obj.resolve())  # 절대 경로로 변환
    logger.info(f"Using data file: {data_path}")
    
    # 캐릭터 이름 추출: generated_agent_dialogue_{character}-{language}.json 형식
    character = 'snow_white'
    if 'generated_agent_dialogue_' in data_path:
        # generated_agent_dialogue_snow_white-korean.json 형식에서 추출
        parts = data_path.split('generated_agent_dialogue_')[1].split('-')
        character = parts[0]
    elif '_' in data_path:
        parts = data_path.split('_')
        for part in parts:
            if 'snow' in part.lower() or 'white' in part.lower():
                character = 'snow_white'
                break
            elif 'mermaid' in part.lower():
                character = 'little_mermaid'
                break
    
    # 프로파일 로드
    seed_data_dir = base_dir / "data" / "seed_data"
    profile_path = seed_data_dir / "profiles" / f"wiki_{character}-korean.txt"
    
    if not profile_path.exists():
        logger.error(f"Profile not found: {profile_path}")
        sys.exit(1)
    
    character_name, _ = read_profile(str(profile_path))
    
    # 메타 프롬프트 로드
    meta_prompt_path = seed_data_dir / "prompts" / "agent_meta_prompt_sft_korean.txt"
    if not meta_prompt_path.exists():
        logger.error(f"Meta prompt not found: {meta_prompt_path}")
        sys.exit(1)
    
    with open(meta_prompt_path, 'r', encoding='utf-8') as fp:
        meta_instruction = fp.read().strip()
    
    # 대화 데이터 로드
    with open(data_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    
    # 출력 경로 설정: trainable-agents 구조에 맞춰 입력 파일과 같은 디렉토리의 prompted/ 폴더에 저장
    input_path = Path(data_path)
    out_path = input_path.parent / "prompted" / f"prompted_agent_dialogue_{character}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sft_data = []
    
    cleaned_count = 0
    too_long_count = 0
    max_output_length = 3000  # trainable-agents 평균(914자)의 약 3배, 너무 긴 응답 방지
    
    for ex in data:
        # 위치/상태 정보는 실제 사용 시 포함되지 않으므로 학습 데이터에서도 제거
        # 실제 사용 프롬프트 형식과 일치시키기 위해 위치/상태 정보 없이 persona_prompt만 사용
        # setting = clean_text(ex.get('background', ''))
        # location = clean_text(ex.get('location', ''))
        
        # 메타 프롬프트는 이제 위치/상태 정보 없이 persona_prompt만 포함
        # format() 호출 없이 그대로 사용 (실제 사용 시와 동일한 형식)
        prompt = meta_instruction
        prompt = clean_text(prompt)
        prompt += '\n\n'
        
        # 대화 텍스트 구성 (<|eot|> 토큰 제거)
        text = ''
        prev_role = ''
        prev_action = ''
        
        for turn in ex.get('dialogue', []):
            role = clean_text(turn.get('role', ''))
            action = clean_text(turn.get('action', ''))
            if not action:
                action = '(말하기)'
            content = clean_text(turn.get('content', ''))
            
            if not content:  # 빈 내용은 건너뛰기
                continue
            
            if text and prev_role == role and prev_action == action:
                text += f'\n{content}'
            else:
                if text:
                    text += '\n'  # <|eot|> 대신 줄바꿈만 사용
                prev_role = role
                prev_action = action
                text += f'{role} {action}: {content}'
        
        # <|eot|> 토큰 제거 (혹시 이미 포함된 경우 대비)
        text = text.replace('<|eot|>', '')
        
        # 정리 후 빈 데이터는 건너뛰기
        if not text.strip():
            cleaned_count += 1
            continue
        
        # 너무 긴 output 필터링 (긴 응답 문제 방지)
        if len(text) > max_output_length:
            too_long_count += 1
            logger.debug(f"Skipping too long output: {len(text)} chars (max: {max_output_length})")
            continue
        
        sft_data.append({
            'prompt': prompt,
            'output': text,
            'source': ex.get('source', 'unknown')
        })
    
    if cleaned_count > 0:
        logger.warning(f"Skipped {cleaned_count} empty entries after cleaning")
    if too_long_count > 0:
        logger.warning(f"Skipped {too_long_count} entries with output longer than {max_output_length} chars")
    
    logger.info(f"Output path: {out_path}")
    logger.info(f"Converted {len(sft_data)} dialogues to training format")
    if sft_data:
        avg_output_length = sum(len(ex['output']) for ex in sft_data) / len(sft_data)
        logger.info(f"Average output length: {avg_output_length:.1f} chars")
    
    # JSONL 형식으로 저장
    with open(out_path, 'w', encoding='utf-8') as fp:
        for ex in sft_data:
            fp.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    # 샘플 출력
    if sft_data:
        logger.info("===SAMPLE INPUT===")
        logger.info(sft_data[0]['prompt'][:500] + "...")
        logger.info("===SAMPLE TARGET===")
        logger.info(sft_data[0]['output'][:500] + "...")
    
    logger.success(f"Saved {len(sft_data)} training samples to {out_path}")


if __name__ == "__main__":
    main()

