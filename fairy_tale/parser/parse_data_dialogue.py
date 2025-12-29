"""
대화 데이터 파싱 스크립트

LLM으로 생성된 대화 데이터를 파싱하여 구조화된 형식으로 변환합니다.
trainable-agents 방식에 맞춰 사용자 질문을 포함한 페르소나 LLM 대화 형식으로 구성합니다.
"""

import json
import os
import sys
from collections import defaultdict
import re
from pathlib import Path
from loguru import logger


def clean_markdown(text: str) -> str:
    """
    마크다운 형식 제거 (**, ##, ### 등)
    
    Args:
        text: 마크다운이 포함된 텍스트
        
    Returns:
        마크다운이 제거된 텍스트
    """
    if not text:
        return ""
    
    # **text** -> text (재귀적으로 처리)
    # 무한 루프 방지: 최대 반복 횟수 제한 및 변경 감지
    max_iterations = 10
    iteration = 0
    
    while '**' in text and iteration < max_iterations:
        # 패턴 1: **text** (완전한 패턴) - 우선 처리
        new_text = re.sub(r'\*\*([^*\n]+?)\*\*', r'\1', text)
        # 패턴 2: **로 시작하는 패턴 (줄 시작 또는 공백 뒤, 줄바꿈 전까지)
        new_text = re.sub(r'(^|\s)\*\*([^*\n]+?)(\s|\n|$)', r'\1\2\3', new_text, flags=re.MULTILINE)
        # 패턴 3: **\n 또는 **\n\n (줄바꿈만 있는 경우)
        new_text = re.sub(r'\*\*\s*\n+', '\n', new_text)
        # 패턴 4: 남은 ** 제거 (단독으로 있는 경우)
        if '**' in new_text:
            new_text = re.sub(r'\*\*', '', new_text)
        
        if new_text == text:  # 변경이 없으면 중단
            break
        text = new_text
        iteration += 1
    
    # ## 제목 -> 제목 (2개 이상)
    text = re.sub(r'^##+\s*', '', text, flags=re.MULTILINE)
    
    # # 제목 -> 제목 (단일 #)
    text = re.sub(r'^#\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#', '', text, flags=re.MULTILINE)
    
    # * 리스트 제거 (단, 문장 중간의 *는 유지)
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)
    
    # (생각), (말하기) 같은 패턴의 괄호는 유지하되, 마크다운 형식만 제거
    # 괄호 안의 내용도 정리
    text = re.sub(r'\(([^)]*?)\)', lambda m: f"({m.group(1).strip()})", text)
    
    # "따옴표" 리터럴 텍스트 제거
    text = re.sub(r'^따옴표\s*\n\s*\n', '', text)
    text = re.sub(r'^따옴표\s*\n', '', text)
    text = re.sub(r'^따옴표\s+', '', text)
    
    return text.strip()


def fix_typos(text: str) -> str:
    """
    일반적인 오타 수정
    
    Args:
        text: 원본 텍스트
        
    Returns:
        오타가 수정된 텍스트
    """
    if not text:
        return text
    
    # 백설공스 → 백설공주 오타 수정
    text = text.replace('백설공스', '백설공주')
    
    return text


def normalize_character_name(name: str, character_name: str = "백설공주") -> str:
    """
    캐릭터 이름 정규화
    - 모호한 이름 (캐릭터1, 캐릭터2 등)을 문맥에 맞게 정규화
    - 백설공주, 왕자, 계모, 난쟁이1 등 명확한 이름으로 변환
    
    Args:
        name: 원본 캐릭터 이름
        character_name: 주인공 캐릭터 이름 (기본값: 백설공주)
        
    Returns:
        정규화된 캐릭터 이름
    """
    if not name:
        return "사용자"
    
    name = name.strip()
    
    # 마크다운 제거
    name = clean_markdown(name)
    
    # 이미 명확한 이름인 경우
    if name in ["백설공주", "사용자", "왕자", "계모", "왕비"]:
        return name
    
    # 난쟁이 관련
    if "난쟁이" in name:
        # 난쟁이1, 난쟁이2 등으로 변환 시도
        num_match = re.search(r'\d+', name)
        if num_match:
            return f"난쟁이{num_match.group()}"
        # "난쟁이들" 같은 복수형도 난쟁이1로
        return "난쟁이1"
    
    # 동물/자연물은 그대로 유지
    if any(keyword in name for keyword in ["새", "나비", "벌", "꽃", "나무", "거울", "시계"]):
        return name
    
    # 모호한 이름 처리
    if name.startswith("캐릭터") or name.startswith("Character"):
        return "사용자"
    
    # 기타는 그대로 반환
    return name


def extract_background(text: str) -> tuple[str, str]:
    """
    배경 텍스트 추출 및 제거 (개선된 버전)
    
    Args:
        text: 원본 텍스트
        
    Returns:
        (배경 텍스트, 나머지 텍스트)
    """
    # 오타 수정 (백설공스 → 백설공주 등)
    text = fix_typos(text)
    
    background = ""
    remaining_text = text
    
    # 배경 패턴 찾기 (더 보수적으로 - 대화 턴 시작 전까지만)
    # 대화 턴 패턴: "사용자 (말하기)", "백설공주 (말하기)" 등
    dialogue_start_pattern = r'(사용자|백설공주|난쟁이\d*|왕자|계모|왕비)\s*\([^)]+\)'
    
    # 배경 패턴 찾기 (다양한 형식 지원)
    patterns = [
        (r'\*\*배경:\*\*\s*\n\s*(.*?)(?=' + dialogue_start_pattern + r'|\*\*|##|$)', re.DOTALL),  # **배경:** 다음 단락
        (r'\*\*배경:\*\*\s*(.*?)(?=' + dialogue_start_pattern + r'|\*\*|##|$)', re.DOTALL),  # **배경:** 같은 줄
        (r'##\s*배경:\s*\n\s*(.*?)(?=' + dialogue_start_pattern + r'|\*\*|##|$)', re.DOTALL),  # ## 배경:
        (r'배경:\s*\n\s*(.*?)(?=' + dialogue_start_pattern + r'|\*\*|##|$)', re.DOTALL),  # 배경: (마크다운 없음)
        (r'\*\*Background:\*\*\s*\n\s*(.*?)(?=' + dialogue_start_pattern + r'|\*\*|##|$)', re.DOTALL),  # 영어 버전
    ]
    
    for pattern, flags in patterns:
        match = re.search(pattern, text, flags)
        if match:
            background = match.group(1).strip()
            # 배경 부분 제거
            remaining_text = text.replace(match.group(0), '').strip()
            break
    
    # 배경 패턴이 없으면 "배경" 단어로 시작하는 첫 단락 찾기
    if not background:
        lines = text.split('\n')
        bg_lines = []
        start_collecting = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # "배경" 또는 "배경:"으로 시작하는 줄 찾기
            if not start_collecting and (line_stripped.startswith('배경') or line_stripped.startswith('**배경') or line_stripped.startswith('## 배경')):
                start_collecting = True
                # "배경:" 뒤의 내용 추출
                if ':' in line_stripped:
                    after_colon = line_stripped.split(':', 1)[1].strip()
                    if after_colon:
                        bg_lines.append(after_colon)
                continue
            
            # 배경 수집 중
            if start_collecting:
                # 대화 턴이 시작되면 중단
                if re.search(dialogue_start_pattern, line_stripped):
                    break
                # 빈 줄은 한 번만 허용
                if not line_stripped:
                    if bg_lines:
                        continue
                    else:
                        break
                bg_lines.append(line_stripped)
        
        if bg_lines:
            background = '\n'.join(bg_lines).strip()
            # 배경 부분 제거 (첫 대화 턴 전까지)
            dialogue_start_match = re.search(dialogue_start_pattern, text)
            if dialogue_start_match:
                bg_end = dialogue_start_match.start()
                remaining_text = text[bg_end:].strip()
            else:
                # 대화 턴이 없으면 배경만 제거
                remaining_text = text.replace(background, '').strip()
    
    # 마크다운 제거
    background = clean_markdown(background)
    
    # 배경이 없으면 첫 단락을 배경으로 추정 (더 보수적으로)
    if not background or len(background) < 30:
        lines = text.split('\n')
        if lines:
            # 첫 번째 비어있지 않은 줄
            first_line = ""
            for line in lines[:5]:  # 더 많은 줄 확인
                line = line.strip()
                if line and not line.startswith('##') and not line.startswith('**'):
                    # 대화 턴 패턴이 포함되어 있으면 배경이 아님
                    if not re.search(dialogue_start_pattern, line):
                        first_line = line
                        break
            
            if first_line and len(first_line) > 50 and not any(keyword in first_line for keyword in ['(', ')', '말하기', '생각']):
                background = clean_markdown(first_line)
                # 첫 줄이 배경이면 제거
                if remaining_text == text:
                    remaining_text = text.replace(first_line, '', 1).strip()
    
    # remaining_text가 비어있거나 너무 짧으면 원본 텍스트 사용
    if not remaining_text or len(remaining_text.strip()) < 50:
        remaining_text = text
    
    return background, remaining_text


def parse_dialogue_turns(text: str, character_name: str = "백설공주") -> list:
    """
    대화 턴 파싱
    
    Args:
        text: 대화 텍스트
        character_name: 주인공 캐릭터 이름
        
    Returns:
        대화 턴 리스트 [{"role": "...", "action": "...", "content": "..."}]
    """
    dialogue = []
    
    if not text or len(text.strip()) < 10:
        return dialogue
    
    # 오타 수정 (백설공스 → 백설공주 등)
    text = fix_typos(text)
    
    # 캐릭터 (행동) 패턴 찾기
    # 패턴 1: **백설공주 (말하기)** 
    # 패턴 2: ## 백설공주 (말하기)
    # 패턴 3: 백설공주 (말하기) (마크다운 없음)
    
    # 패턴 정의: 더 구체적인 패턴부터 처리
    patterns = [
        r'\*\*([^*]+?)\s*\(([^)]+?)\)\*\*',  # **캐릭터 (행동)** - 가장 구체적
        r'##\s*([^(]+?)\s*\(([^)]+?)\)',  # ## 캐릭터 (행동)
        r'^([^\n(]+?)\s*\(([^)]+?)\)',  # 캐릭터 (행동) (줄 시작) - 가장 일반적
    ]
    
    all_matches = []
    used_ranges = []  # 이미 사용된 범위 저장
    
    # 패턴을 순차적으로 처리 (더 구체적인 패턴부터)
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        for m in matches:
            start, end = m.start(), m.end()
            # 이미 사용된 범위와 겹치지 않는 경우만 추가
            is_overlapping = any(start < used_end and end > used_start 
                               for used_start, used_end in used_ranges)
            if not is_overlapping:
                all_matches.append((start, end, m.group(1), m.group(2)))
                used_ranges.append((start, end))
    
    # 시작 위치 기준으로 정렬
    all_matches.sort(key=lambda x: x[0])
    
    if not all_matches:
        return dialogue
    
    # 각 매치 사이의 내용을 추출
    for i, (start_pos, end_pos, role, action) in enumerate(all_matches):
        role = role.strip()
        action = action.strip()
        
        # 다음 매치까지의 내용 추출
        content_start = end_pos
        if i + 1 < len(all_matches):
            content_end = all_matches[i + 1][0]
        else:
            content_end = len(text)
        
        content = text[content_start:content_end].strip()
        
        # 오타 수정
        role = fix_typos(role)
        action = fix_typos(action)
        content = fix_typos(content)
        
        # 마크다운 제거 (강화)
        role = clean_markdown(role)
        action = clean_markdown(action)
        content = clean_markdown(content)
        
        # 추가 마크다운 제거: 모든 **, *, # 패턴 제거
        # role 정리
        role = re.sub(r'^#+\s*', '', role)  # 시작 부분의 # 제거
        role = re.sub(r'^\*\*+', '', role)  # 시작 부분의 ** 제거
        role = re.sub(r'\*\*+$', '', role)  # 끝 부분의 ** 제거
        role = re.sub(r'^\*+', '', role)  # 시작 부분의 * 제거
        role = role.strip()
        
        # Role 품질 검증 및 필터링
        # 1. 메타데이터 제거 (더 강화된 패턴)
        # "대화 턴 N", "턴 N", "턴 수: N개" 등 모든 메타데이터 제거
        role = re.sub(r'^대화\s*턴\s*\d+\s*', '', role, flags=re.IGNORECASE)
        role = re.sub(r'^턴\s*수\s*:\s*\d+\s*개\s*(이상\s*)?', '', role, flags=re.IGNORECASE)
        role = re.sub(r'^턴\s*시작\s*:?\s*', '', role, flags=re.IGNORECASE)
        role = re.sub(r'^턴\s*:?\s*', '', role, flags=re.IGNORECASE)
        role = re.sub(r'^대화\s*시작\s*', '', role, flags=re.IGNORECASE)
        role = re.sub(r'^대화\s*', '', role, flags=re.IGNORECASE)
        role = re.sub(r'^첫\s*번째\s*대화\s*', '', role, flags=re.IGNORECASE)
        role = re.sub(r'^추가\s*대화\s*예시\s*:?\s*', '', role, flags=re.IGNORECASE)
        role = re.sub(r'\n+', ' ', role)  # 줄바꿈을 공백으로 변환
        role = role.strip()
        
        # 2. 앞뒤 콜론/특수문자 제거
        role = re.sub(r'^:\s*', '', role)  # 앞의 콜론 제거
        role = re.sub(r'\s*:\s*$', '', role)  # 뒤의 콜론 제거
        role = role.strip()
        
        # 3. 배경 설명이 role에 포함된 경우 필터링
        # 배경 설명 키워드가 포함되어 있고 길이가 50자 이상이면 배경으로 판단
        background_keywords = ['배경', '맑은', '햇살', '숲', '계곡', '풍경', '아름다운', '고요한']
        if any(keyword in role for keyword in background_keywords) and len(role) > 50:
            # 배경 설명으로 판단되면 건너뛰기
            continue
        
        # 4. Role 길이 제한 (150자 이상이면 경고하고 건너뛰기) - 100자에서 150자로 완화
        if len(role) > 150:
            logger.warning(f"Role too long ({len(role)} chars), skipping: {role[:50]}...")
            continue
        
        # 5. 특수문자만 있는 경우 필터링 (예: "\:", ":")
        if role in ['\\:', ':', '\\', '/', '']:
            continue
        
        # content 정리
        # "따옴표\n\n", "따옴표\n", "따옴표" 패턴 제거
        content = re.sub(r'^따옴표\s*\n\s*\n', '', content)
        content = re.sub(r'^따옴표\s*\n', '', content)
        content = re.sub(r'^따옴표\s+', '', content)
        
        content = re.sub(r'^\*\*\s*\n+', '', content)  # **\n 제거
        content = re.sub(r'^\*\*+', '', content)  # 시작 부분의 ** 제거
        content = re.sub(r'^\*\s+', '', content)  # * 리스트 마커 제거
        content = re.sub(r'\*\*+', '', content)  # 모든 남은 ** 제거
        content = re.sub(r'^\*+', '', content)  # 시작 부분의 * 제거
        content = content.strip()
        
        # Content 품질 검증 및 필터링
        # 1. 최소 길이 검증 완화 (10자 → 5자)
        if not content or len(content) < 5:
            continue
        
        # 2. 플레이스홀더 패턴 감지 및 필터링 (완화)
        # 정확히 일치하는 패턴만 필터링 (content에 "상세한 발언"이 포함된 것은 정상)
        placeholder_patterns = [
            r'^상세한\s*발언\s*\.?\s*$',  # 정확히 "상세한 발언" 또는 "상세한 발언."만
            r'^내면의\s*생각\s*\.?\s*$',  # 정확히 "내면의 생각" 또는 "내면의 생각."만
            r'^백설공주\s*\.?\s*$',  # 정확히 "백설공주" 또는 "백설공주."만
            r'^\("백설공주"\)\s*$',  # ("백설공주")만
        ]
        is_placeholder = False
        for pattern in placeholder_patterns:
            if re.match(pattern, content.strip(), re.IGNORECASE):
                is_placeholder = True
                break
        
        # "상세한 발언 ..." 같은 패턴은 필터링하지 않음 (내용이 있으면 유지)
        if is_placeholder and len(content.strip()) < 20:
            continue
        
        # 캐릭터 이름 정규화
        role = normalize_character_name(role, character_name)
        
        # action 정규화 (말하기/생각)
        if "생각" in action or "thinking" in action.lower():
            action = "생각"
        else:
            action = "말하기"
        
        dialogue.append({
            "role": role,
            "action": action,
            "content": content
        })
    
    return dialogue


def add_user_questions(dialogue: list, character_name: str = "백설공주", scene_type: str = "", location: str = "", background: str = "") -> list:
    """
    사용자 질문 추가
    페르소나 LLM 대화 형식을 위해 사용자가 백설공주에게 질문하는 턴을 추가
    
    Args:
        dialogue: 기존 대화 리스트
        character_name: 주인공 캐릭터 이름
        scene_type: 장면 유형
        location: 장면 장소
        background: 장면 배경
        
    Returns:
        사용자 질문이 추가된 대화 리스트
    """
    if not dialogue:
        return dialogue
    
    new_dialogue = []
    
    # 첫 번째 턴이 사용자 질문이 아니면 추가
    first_turn = dialogue[0]
    
    if first_turn["role"] != "사용자":
        # 장면 정보를 바탕으로 자연스러운 사용자 질문 생성
        question_parts = []
        
        if location:
            question_parts.append(f"{location}에서")
        if scene_type:
            question_parts.append(f"{scene_type} 상황에 대해")
        if background:
            bg_short = background[:80] + "..." if len(background) > 80 else background
            question_parts.append(f"이 상황({bg_short})")
        
        if question_parts:
            question = " ".join(question_parts) + " 이야기해줄래?"
        else:
            question = "이 상황에 대해 이야기해줄래?"
        
        new_dialogue.append({
            "role": "사용자",
            "action": "말하기",
            "content": question
        })
    
    # 기존 대화 추가
    for turn in dialogue:
        # 주인공 캐릭터가 아닌 다른 캐릭터의 발언은 주인공의 설명으로 변환하거나 유지
        # 단, 사용자는 그대로 유지
        if turn["role"] == "사용자":
            new_dialogue.append(turn)
        elif turn["role"] == character_name:
            new_dialogue.append(turn)
        else:
            # 다른 캐릭터의 발언도 포함 (난쟁이, 동물 등)
            new_dialogue.append(turn)
    
    return new_dialogue


def load_gen_data(path: str) -> list:
    """
    생성된 데이터 로드 (JSONL 형식 - 한 줄씩 읽기)
    
    Args:
        path: JSONL 파일 경로
        
    Returns:
        데이터 리스트
    """
    data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line_num, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                data.append(ex)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                continue
    return data


def parse_dialogue_info(text: str, character_name: str = "백설공주", scene_type: str = "", location: str = "", scene_background: str = "") -> dict | str:
    """
    대화 정보 파싱 (개선된 버전)
    
    Args:
        text: 대화 텍스트
        character_name: 주인공 캐릭터 이름
        scene_type: 장면 유형
        location: 장면 장소
        scene_background: 장면 배경 설명
        
    Returns:
        파싱된 대화 정보 또는 'INV' (잘못된 형식)
    """
    if not text or len(text.strip()) < 50:
        return 'INV'
    
    # 예시 형식 제거
    if '예시 형식:' in text:
        text = text.split('예시 형식:')[-1]
    if 'Example format:' in text:
        text = text.split('Example format:')[-1]
    
    # 배경 추출
    background, remaining_text = extract_background(text)
    
    # 배경이 없거나 너무 짧으면 scene_background 사용
    if not background or len(background) < 30:
        background = scene_background
    
    # 대화 턴 파싱
    # remaining_text가 비어있거나 너무 짧으면 원본 텍스트 사용
    if not remaining_text or len(remaining_text.strip()) < 50:
        remaining_text = text
    
    dialogue = parse_dialogue_turns(remaining_text, character_name)
    
    if not dialogue:
        # 대화 턴이 없으면 원본 텍스트에서 직접 파싱 시도
        dialogue = parse_dialogue_turns(text, character_name)
        if not dialogue:
            return 'INV'
    
    # 사용자 질문 추가 (페르소나 LLM 형식)
    dialogue = add_user_questions(dialogue, character_name, scene_type, location, background)
    
    return {
        'background': background,
        'dialogue': dialogue
    }


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        logger.error("Usage: python parse_data_dialogue.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    # 경로 처리: 여러 가능한 경로 시도
    base_dir = Path(__file__).parent.parent  # fairy_tale/
    project_root = base_dir.parent  # 프로젝트 루트
    current_dir = Path.cwd()
    
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
            base_dir / "result" / Path(data_path).relative_to(Path("result")) if str(data_path).startswith("result/") else None,  # result/ 제거 후 fairy_tale/result/ 추가
        ]
        
        # None 제거
        possible_paths = [p for p in possible_paths if p is not None]
        
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
        if str(data_path).startswith("result/"):
            logger.info(f"  - {base_dir / 'result' / Path(data_path).relative_to(Path('result'))}")
        sys.exit(1)
    
    data_path = str(data_path_obj.resolve())  # 절대 경로로 변환
    logger.info(f"Using data file: {data_path}")
    
    # 캐릭터 이름 추출
    character = 'snow_white'  # 기본값
    if '-char-' in data_path:
        character = data_path.split('-char-')[1].split('-')[0]
    
    # 언어 추출
    language = 'korean'  # 기본값
    if '-korean' in data_path:
        language = 'korean'
    elif '-english' in data_path:
        language = 'english'
    
    # 캐릭터 이름 매핑 (영문 -> 한글)
    character_names = {
        'snow_white': '백설공주',
        'little_mermaid': '인어공주',
    }
    character_name = character_names.get(character, '백설공주')
    
    # 날짜 추출: 파싱을 실행한 현재 날짜 사용
    from datetime import datetime
    parse_date = datetime.now().strftime('%Y-%m-%d')
    
    # 출력 경로 설정: trainable-agents 구조에 맞춰서
    out_path = base_dir / "processed" / parse_date / f"generated_agent_dialogue_{character}-{language}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    counter = defaultdict(int)
    raw_data = load_gen_data(data_path)
    results = []
    
    logger.info(f"Loading {len(raw_data)} raw dialogue data entries...")
    
    for idx, ex in enumerate(raw_data, 1):
        if not ex.get('check_result', False):
            counter['skip_no_check'] += 1
            continue
        
        gen_id = ex.get('gen_answer_id', 'unknown')
        
        # 장면 데이터 추출
        scene_data = ex.get('scene_data', {})
        loc = scene_data.get('location', '')
        scene_type = scene_data.get('type', '')
        back = scene_data.get('background', '')
        
        # 완성된 텍스트에서 대화 추출
        generated = ex.get('completions', '')
        
        if not generated or len(generated.strip()) < 50:
            counter['empty'] += 1
            if idx <= 5:
                logger.warning(f"Empty or too short completions for {gen_id}")
            continue
        
        # 대화 파싱
        out = parse_dialogue_info(generated, character_name, scene_type, loc, back)
        
        if isinstance(out, str):
            counter[out] += 1
            if idx <= 5:  # 처음 5개만 자세히 로깅
                logger.warning(f"Failed to parse dialogue for {gen_id}: {out}")
            continue
        
        # 장면 정보 추가
        out['location'] = loc
        if not out.get('background') or len(out.get('background', '')) < 30:
            out['background'] = back
        
        out['source'] = f'seed_dialogue_{gen_id}'
        
        results.append(out)
    
    logger.info(f"Output path: {out_path}")
    logger.info(f"Successfully parsed {len(results)}/{len(raw_data)} dialogues")
    logger.info(f"Invalid/empty dialogues: {dict(counter)}")
    
    # 통계 출력
    if results:
        sum_turns = 0
        sum_turn_words = 0
        user_turns = 0
        character_turns = 0
        
        for r in results:
            dialogue = r.get('dialogue', [])
            sum_turns += len(dialogue)
            
            for d in dialogue:
                content = d.get('content', '')
                sum_turn_words += len(content.split())
                
                if d.get('role') == '사용자':
                    user_turns += 1
                elif d.get('role') == character_name:
                    character_turns += 1
        
        logger.info(
            f"Total turns: {sum_turns}, "
            f"Avg turns per dialogue: {sum_turns/len(results):.2f}, "
            f"User turns: {user_turns}, Character turns: {character_turns}"
        )
    
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    
    logger.success(f"Saved {len(results)} dialogues to {out_path}")


if __name__ == "__main__":
    main()
