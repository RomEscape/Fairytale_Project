"""
대화 데이터에서 마크다운 제거 스크립트

생성된 대화 데이터의 completions 필드에서 마크다운 형식을 제거합니다.
"""

import json
import re
import sys
from pathlib import Path
from tqdm import tqdm


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
    
    # ## 제목 -> 제목 (2개 이상, 줄 시작 또는 공백 뒤)
    text = re.sub(r'(^|\s)##+\s*', r'\1', text, flags=re.MULTILINE)
    # 남은 ## 패턴 제거 (어디에 있든)
    text = re.sub(r'##+', '', text)
    
    # # 제목 -> 제목 (단일 #, 줄 시작 또는 공백 뒤)
    text = re.sub(r'(^|\s)#\s+', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'(^|\s)#', r'\1', text, flags=re.MULTILINE)
    
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


def clean_dialogue_file(input_path: Path, output_path: Path = None) -> None:
    """
    대화 파일에서 마크다운 제거
    
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
    backup_path = input_path.with_suffix(input_path.suffix + ".backup")
    print(f"백업 생성: {backup_path}")
    import shutil
    shutil.copy2(input_path, backup_path)
    
    print(f"파일 읽기: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"전체 대화 수: {len(lines):,}개")
    
    # 마크다운이 있는 대화 확인
    markdown_count = 0
    for line in lines:
        try:
            data = json.loads(line)
            completions = data.get("completions", "")
            if "**" in completions or "##" in completions or "###" in completions or ("# " in completions and completions.count("#") > 5):
                markdown_count += 1
        except:
            pass
    
    print(f"마크다운이 있는 대화: {markdown_count:,}개 ({markdown_count/len(lines)*100:.2f}%)")
    print(f"")
    print(f"마크다운 제거 시작...")
    
    cleaned_count = 0
    cleaned_lines = []
    
    # Progress bar
    progress_bar = tqdm(
        total=len(lines),
        desc="Cleaning markdown",
        unit="dialogue",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for line in lines:
        try:
            data = json.loads(line)
            original_completions = data.get("completions", "")
            
            # 마크다운 제거
            cleaned_completions = clean_markdown(original_completions)
            
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
    print(f"   - 마크다운 제거된 대화: {cleaned_count:,}개")
    print(f"   - 백업 파일: {backup_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_dialogue_markdown.py <dialogue_file_path> [output_file_path]")
        print("")
        print("예시:")
        print("  python clean_dialogue_markdown.py result/2025-12-10/gen_dialogue/dialogue_...jsonl")
        print("  python clean_dialogue_markdown.py input.jsonl output.jsonl")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    clean_dialogue_file(input_path, output_path)

