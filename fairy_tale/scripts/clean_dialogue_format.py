"""
대화 데이터 포맷 정리 스크립트

띄어쓰기, 줄바꿈 등 포맷 문제를 정리합니다.
"""

import json
import re
import sys
from pathlib import Path
from tqdm import tqdm


def clean_formatting(text: str) -> str:
    """
    포맷 정리 (띄어쓰기, 줄바꿈 등)
    
    Args:
        text: 정리할 텍스트
        
    Returns:
        정리된 텍스트
    """
    if not text:
        return ""
    
    # 연속 공백 (3개 이상) -> 1개로 변경
    text = re.sub(r' {3,}', ' ', text)
    
    # 연속 줄바꿈 (3개 이상) -> 2개로 변경
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 탭 문자 -> 공백으로 변경
    text = text.replace('\t', ' ')
    
    # 줄 끝의 공백 제거
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)
    
    # 연속 공백 (2개) -> 1개로 변경 (단, 줄 시작의 공백은 유지)
    # 줄 중간의 연속 공백만 정리
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # 줄 시작 공백은 유지하고, 중간의 연속 공백만 정리
        leading_spaces = len(line) - len(line.lstrip())
        content = line.lstrip()
        # 중간의 연속 공백 정리
        content = re.sub(r' {2,}', ' ', content)
        cleaned_lines.append(' ' * leading_spaces + content if leading_spaces > 0 else content)
    text = '\n'.join(cleaned_lines)
    
    # 제어 문자 제거 (일반 텍스트에 사용되지 않는 제어 문자)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    return text.strip()


def clean_dialogue_file(input_path: Path, output_path: Path = None) -> None:
    """
    대화 파일 포맷 정리
    
    Args:
        input_path: 입력 파일 경로
        output_path: 출력 파일 경로 (None이면 입력 파일에 덮어쓰기)
    """
    if not input_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {input_path}")
        return
    
    if output_path is None:
        output_path = input_path
    
    # 백업 생성
    backup_path = input_path.with_suffix(input_path.suffix + ".backup_format")
    print(f"백업 생성: {backup_path}")
    import shutil
    shutil.copy2(input_path, backup_path)
    
    print(f"파일 읽기: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"전체 대화 수: {len(lines):,}개")
    print(f"")
    print(f"포맷 정리 시작...")
    
    cleaned_count = 0
    cleaned_lines = []
    
    # Progress bar
    progress_bar = tqdm(
        total=len(lines),
        desc="Cleaning format",
        unit="dialogue",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for line in lines:
        try:
            data = json.loads(line)
            original_completions = data.get("completions", "")
            
            # 포맷 정리
            cleaned_completions = clean_formatting(original_completions)
            
            if cleaned_completions != original_completions:
                cleaned_count += 1
                data["completions"] = cleaned_completions
            
            cleaned_lines.append(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"\n경고: 라인 파싱 오류: {e}")
            cleaned_lines.append(line)  # 오류가 나도 원본 유지
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    # 파일 저장
    print(f"")
    print(f"파일 저장: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)
    
    print(f"완료!")
    print(f"   - 처리된 대화: {len(lines):,}개")
    print(f"   - 포맷 정리된 대화: {cleaned_count:,}개")
    print(f"   - 백업 파일: {backup_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_dialogue_format.py <dialogue_file_path> [output_file_path]")
        print("")
        print("예시:")
        print("  python clean_dialogue_format.py result/2025-12-10/gen_dialogue/dialogue_...jsonl")
        print("  python clean_dialogue_format.py input.jsonl output.jsonl")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    clean_dialogue_file(input_path, output_path)

