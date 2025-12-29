#!/usr/bin/env python3
"""
캐릭터 전환 스크립트
conf.yaml의 character_config.conf_name을 변경하여 캐릭터를 전환합니다.
"""
import sys
import os
import yaml
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
CONF_FILE = PROJECT_ROOT / "conf.yaml"
CHARACTERS_DIR = PROJECT_ROOT / "characters"


def list_characters() -> list[str]:
    """사용 가능한 캐릭터 목록 반환"""
    characters = []
    if CHARACTERS_DIR.exists():
        for yaml_file in CHARACTERS_DIR.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if "character_config" in data:
                        conf_name = data["character_config"].get("conf_name", yaml_file.stem)
                        characters.append((yaml_file.stem, conf_name))
            except Exception as e:
                print(f"⚠️  캐릭터 파일 {yaml_file.name} 로드 실패: {e}", file=sys.stderr)
    return characters


def get_current_character() -> str | None:
    """현재 설정된 캐릭터 이름 반환"""
    if not CONF_FILE.exists():
        return None

    try:
        with open(CONF_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if "character_config" in data:
                return data["character_config"].get("conf_name")
    except Exception as e:
        print(f"⚠️  설정 파일 로드 실패: {e}", file=sys.stderr)
    return None


def switch_character(character_name: str) -> bool:
    """캐릭터 전환"""
    if not CONF_FILE.exists():
        print(f"❌ 설정 파일을 찾을 수 없습니다: {CONF_FILE}", file=sys.stderr)
        return False

    # 캐릭터 파일 확인
    character_file = CHARACTERS_DIR / f"{character_name}.yaml"
    if not character_file.exists():
        print(f"❌ 캐릭터 파일을 찾을 수 없습니다: {character_file}", file=sys.stderr)
        print(f"💡 사용 가능한 캐릭터: {', '.join([c[0] for c in list_characters()])}")
        return False

    # 캐릭터 파일에서 conf_name 읽기
    try:
        with open(character_file, "r", encoding="utf-8") as f:
            char_data = yaml.safe_load(f)
            if "character_config" not in char_data:
                print(f"❌ 캐릭터 파일 형식이 올바르지 않습니다: {character_file}", file=sys.stderr)
                return False
            target_conf_name = char_data["character_config"].get("conf_name", character_name)
    except Exception as e:
        print(f"❌ 캐릭터 파일 로드 실패: {e}", file=sys.stderr)
        return False

    # conf.yaml 읽기
    try:
        with open(CONF_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            data = yaml.safe_load(content)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}", file=sys.stderr)
        return False

    # character_config.conf_name 변경
    if "character_config" not in data:
        data["character_config"] = {}

    old_conf_name = data["character_config"].get("conf_name", "알 수 없음")
    data["character_config"]["conf_name"] = target_conf_name

    # conf.yaml 저장
    try:
        # 원본 백업
        backup_file = PROJECT_ROOT / "conf.yaml.backup"
        if CONF_FILE.exists():
            import shutil

            shutil.copy2(CONF_FILE, backup_file)
            print(f"💾 백업 생성: {backup_file}")

        with open(CONF_FILE, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        print(f"✅ 캐릭터 전환 완료!")
        print(f"   이전: {old_conf_name}")
        print(f"   현재: {target_conf_name}")
        print(f"   캐릭터 파일: {character_file.name}")
        return True
    except Exception as e:
        print(f"❌ 설정 파일 저장 실패: {e}", file=sys.stderr)
        return False


def main():
    if len(sys.argv) < 2:
        print("📋 사용법:")
        print(f"  {sys.argv[0]} <캐릭터_이름>")
        print(f"  {sys.argv[0]} --list")
        print()
        print("예시:")
        print(f"  {sys.argv[0]} snow_white")
        print()

        current = get_current_character()
        if current:
            print(f"현재 캐릭터: {current}")
        print()

        characters = list_characters()
        if characters:
            print("📚 사용 가능한 캐릭터:")
            for file_name, conf_name in characters:
                marker = "👉 " if conf_name == current else "   "
                print(f"{marker}{file_name} ({conf_name})")
        sys.exit(0)

    if sys.argv[1] == "--list":
        characters = list_characters()
        if characters:
            print("📚 사용 가능한 캐릭터:")
            for file_name, conf_name in characters:
                print(f"  - {file_name} ({conf_name})")
        else:
            print("❌ 사용 가능한 캐릭터가 없습니다.")
        sys.exit(0)

    character_name = sys.argv[1]
    success = switch_character(character_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

