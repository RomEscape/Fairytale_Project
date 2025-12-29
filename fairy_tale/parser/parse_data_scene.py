"""
장면 데이터 파싱 스크립트

LLM으로 생성된 장면 데이터를 파싱하여 구조화된 형식으로 변환합니다.
참고 프로젝트의 parse_data_scene.py 구조를 따릅니다.
"""

import json
import os
import sys
from collections import defaultdict
import re
from pathlib import Path
from loguru import logger


def load_gen_data(path: str) -> list:
    """
    생성된 데이터 로드 (JSONL 형식)
    
    Args:
        path: JSONL 파일 경로
    
    Returns:
        데이터 리스트
    """
    with open(path, 'r', encoding='utf-8') as fp:
        raw = fp.read().split('}\n{')
    data = []
    for s in raw:
        s = s.strip()
        if not s.startswith('{'):
            s = '{' + s
        if not s.endswith('}'):
            s = s + '}'
        try:
            ex = json.loads(s)
            data.append(ex)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            continue
    return data


def clean_markdown(text: str) -> str:
    """
    마크다운 형식 제거
    
    Args:
        text: 원본 텍스트
        
    Returns:
        마크다운이 제거된 텍스트
    """
    if not text:
        return ''
    
    # 마크다운 헤더 제거 (###, ##, #)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # 마크다운 볼드/이탤릭 제거 (**text**, *text*)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **text** -> text
    text = re.sub(r'\*([^*\n]+)\*', r'\1', text)  # *text* -> text
    
    # 마크다운 리스트 기호 제거 (*, -, +)
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    
    # 남은 마크다운 기호 제거
    text = text.replace('**', '').replace('*', '').strip()
    
    return text


def extract_field_value(text: str, field_name: str) -> tuple[str | None, int]:
    """
    텍스트에서 필드 값을 추출 (정규식 사용, 다양한 형식 지원)
    
    Args:
        text: 원본 텍스트
        field_name: 필드 이름 ('유형', '장소', '배경' 등)
        
    Returns:
        (필드 값, 발견된 줄 인덱스) 또는 (None, -1)
    """
    lines = text.split('\n')
    
    # 다양한 필드 패턴 지원
    patterns = [
        rf'{field_name}\s*:\s*(.+)',  # 유형: 값
        rf'\*\*{field_name}\*\*\s*:\s*(.+)',  # **유형**: 값
        rf'\*{field_name}\*\s*:\s*(.+)',  # *유형*: 값
        rf'{field_name}\s*:\s*\*\*(.+?)\*\*',  # 유형: **값**
        rf'{field_name}\s*:\s*\*(.+?)\*',  # 유형: *값*
    ]
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                value = clean_markdown(value)
                if value:
                    return value, i
    
    return None, -1


def extract_background_multiline(text: str, start_line_idx: int) -> str:
    """
    배경 필드의 여러 줄 내용 추출
    
    Args:
        text: 원본 텍스트
        start_line_idx: 배경 필드가 시작되는 줄 인덱스
        
    Returns:
        추출된 배경 내용
    """
    lines = text.split('\n')
    bg_lines = []
    
    # 배경 필드가 시작된 줄에서 ':' 뒤 내용 추출
    if start_line_idx < len(lines):
        start_line = lines[start_line_idx]
        if ':' in start_line:
            after_colon = start_line.split(':', 1)[1].strip()
            after_colon = clean_markdown(after_colon)
            if after_colon:
                bg_lines.append(after_colon)
    
    # 다음 줄들에서 배경 내용 수집
    for i in range(start_line_idx + 1, len(lines)):
        line = lines[i].strip()
        
        # 다음 필드/장면 시작 시 중단
        if re.search(r'장면\s*\d+\s*:', line, re.IGNORECASE):
            # 장면 번호인 경우 중단
            break
        if re.search(r'유형\s*:', line, re.IGNORECASE):
            break
        if re.search(r'장소\s*:', line, re.IGNORECASE):
            break
        
        # 빈 줄은 한 번만 허용
        if not line:
            if bg_lines:
                continue
            else:
                continue
        
        # 마크다운 제거 후 추가
        cleaned_line = clean_markdown(line)
        if cleaned_line:
            bg_lines.append(cleaned_line)
    
    return ' '.join(bg_lines).strip()


def parse_scene_info(text: str) -> dict | str:
    """
    장면 정보 파싱 (마크다운 및 단순 텍스트 형식 모두 지원)
    
    지원하는 형식:
    - 단순 텍스트: 유형: 대화
    - 마크다운 볼드: **유형**: 대화
    - 마크다운 리스트: *유형: 대화
    - 마크다운 헤더: ### 장면 1:
    
    Args:
        text: 장면 텍스트
    
    Returns:
        파싱된 장면 정보 또는 'INV' (잘못된 형식)
    """
    if not text or not text.strip():
        return 'INV'
    
    res = {}
    
    # 유형 추출
    type_value, type_line_idx = extract_field_value(text, '유형')
    if not type_value:
        # Type: (영문) 시도
        type_value, type_line_idx = extract_field_value(text, 'Type')
    
    if type_value:
        res['type'] = type_value
    else:
        return 'INV'
    
    # 장소 추출
    location_value, location_line_idx = extract_field_value(text, '장소')
    if not location_value:
        # Location: (영문) 시도
        location_value, location_line_idx = extract_field_value(text, 'Location')
    
    if location_value:
        res['location'] = location_value
    else:
        return 'INV'
    
    # 배경 추출 (여러 줄 지원)
    background_value, background_line_idx = extract_field_value(text, '배경')
    if not background_value:
        # Background: (영문) 시도
        background_value, background_line_idx = extract_field_value(text, 'Background')
    
    if background_line_idx >= 0:
        # 여러 줄 배경 추출
        multiline_bg = extract_background_multiline(text, background_line_idx)
        if multiline_bg:
            res['background'] = multiline_bg
        elif background_value:
            res['background'] = background_value
        else:
            return 'INV'
    elif background_value:
        res['background'] = background_value
    else:
        return 'INV'
    
    # 모든 필수 필드 확인
    if not all(k in res for k in ['type', 'location', 'background']):
        return 'INV'
    
    if not all(res[k].strip() for k in ['type', 'location', 'background']):
        return 'INV'
    
    return res


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        logger.error("Usage: python parse_data_scene.py <data_path>")
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
        character = data_path.split('-char-')[1].split('-')[0]  # 예: snow_white-korean.jsonl -> snow_white
    
    # 언어 추출
    language = 'korean'  # 기본값
    if '-korean' in data_path:
        language = 'korean'
    elif '-english' in data_path:
        language = 'english'
    
    # 날짜 추출: 파싱을 실행한 현재 날짜 사용
    # (원본 데이터가 언제 생성되었는지와 관계없이 파싱 실행 날짜 기준)
    from datetime import datetime
    parse_date = datetime.now().strftime('%Y-%m-%d')
    
    # 출력 경로 설정: trainable-agents 구조에 맞춰서
    # 파싱 날짜 디렉토리에 저장 (원본 데이터 날짜와 무관)
    out_path = base_dir / "processed" / parse_date / f"generated_agent_scene_{character}-{language}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    counter = defaultdict(int)
    raw_data = load_gen_data(data_path)
    results = []
    
    for ex in raw_data:
        if not ex.get('check_result', False):
            continue
        
        gen_id = ex.get('gen_answer_id', 'unknown')
        cid = 1
        
        # 프로파일 추출
        prompt = ex.get('prompt', '')
        profile = ''
        if '맥락:' in prompt:
            profile = prompt.split('맥락:')[1].split('위의 맥락')[0].strip()
        elif 'Context:' in prompt:
            profile = prompt.split('Context:')[1].split('Imagine')[0].strip()
        
        # 완성된 텍스트에서 장면 추출
        completions = ex.get('completions', '')
        
        # 장면 분할: 다양한 형식 지원
        # 패턴: "장면 X:", "### 장면 X:", "## 장면 X:", "**장면 X:**" 등
        scene_pattern = r'(?:^|\n)(?:#{1,6}\s*)?(?:\*\*)?\s*장면\s*(\d+)\s*(?:\*\*)?\s*:'
        scene_matches = list(re.finditer(scene_pattern, completions, re.MULTILINE | re.IGNORECASE))
        
        scenes = []
        if scene_matches:
            # 각 장면의 시작 위치 찾기
            for i, match in enumerate(scene_matches):
                start_pos = match.start()
                # 다음 장면 시작 위치 또는 텍스트 끝
                if i + 1 < len(scene_matches):
                    end_pos = scene_matches[i + 1].start()
                else:
                    end_pos = len(completions)
                
                scene_text = completions[start_pos:end_pos].strip()
                if scene_text:
                    scenes.append(scene_text)
        else:
            # 정규식으로 찾지 못한 경우, 기존 방식 사용 (빈 줄로 분할)
            scenes = completions.split('\n\n')
        
        for scene_text in scenes:
            if not scene_text.strip():
                continue
            
            # '장면' 키워드가 없으면 건너뜀 (헤더 제외하고 장면 내용 확인)
            if '장면' not in scene_text and '유형' not in scene_text and 'Type' not in scene_text:
                continue
            
            out = parse_scene_info(scene_text)
            if isinstance(out, str):
                counter[out] += 1
                continue
            
            out['source'] = f'seed_scene_{gen_id}_c{cid}'
            out['profile'] = profile
            cid += 1
            results.append(out)
    
    logger.info(f"Output path: {out_path}")
    logger.info(f"Parsed {len(results)} scenes")
    logger.info(f"Invalid scenes: {dict(counter)}")
    
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    
    logger.success(f"Saved {len(results)} scenes to {out_path}")


if __name__ == "__main__":
    main()

