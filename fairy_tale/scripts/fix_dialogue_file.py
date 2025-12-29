"""
잘못 생성된 대화 파일 수정 스크립트

1356번째 장면의 21번째 대화부터 잘못 생성된 부분을 삭제하고,
원래 끊긴 지점(1356번째 장면의 21번째 대화)부터 다시 생성할 수 있도록 합니다.
"""

import json
from pathlib import Path


def fix_dialogue_file(
    dialogue_file_path: Path,
    dialogues_per_scene: int = 35,
    target_scene: int = 1356,
) -> None:
    """
    잘못 생성된 대화 파일 수정
    
    Args:
        dialogue_file_path: 대화 파일 경로
        dialogues_per_scene: 각 장면당 대화 개수
        target_scene: 수정할 장면 번호 (1-based)
    """
    if not dialogue_file_path.exists():
        print(f"오류: File not found: {dialogue_file_path}")
        return
    
    # 1356번째 장면의 20번째 대화까지는 정상
    target_scene_idx = target_scene - 1  # 0-based
    correct_end_line = target_scene_idx * dialogues_per_scene + 19  # 20번째 대화 (0-based: 19)
    
    print(f"Reading file: {dialogue_file_path}")
    with open(dialogue_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines):,}")
    print(f"Correct end line (1356번째 장면의 20번째 대화): {correct_end_line + 1}")
    
    if len(lines) <= correct_end_line + 1:
        print("File is already correct. No fix needed.")
        return
    
    # 잘못 생성된 부분 삭제
    lines_to_keep = lines[:correct_end_line + 1]
    lines_to_remove = len(lines) - len(lines_to_keep)
    
    print(f"경고: Removing {lines_to_remove} incorrectly generated lines (line {correct_end_line + 2} ~ {len(lines)})")
    
    # 백업 생성
    backup_path = dialogue_file_path.with_suffix(dialogue_file_path.suffix + ".backup")
    print(f"Creating backup: {backup_path}")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    # 수정된 파일 저장
    print(f"Writing corrected file: {dialogue_file_path}")
    with open(dialogue_file_path, "w", encoding="utf-8") as f:
        f.writelines(lines_to_keep)
    
    print(f"Fixed file: {len(lines_to_keep):,} lines kept, {lines_to_remove} lines removed")
    print(f"File is now ready to resume from scene {target_scene}, dialogue 21")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fix_dialogue_file.py <dialogue_file_path>")
        sys.exit(1)
    
    dialogue_file_path = Path(sys.argv[1])
    fix_dialogue_file(dialogue_file_path)

