#!/usr/bin/env python3
"""
HuggingFace Hub에 모델 파일 업로드 스크립트

사용법:
    python fairy_tale/scripts/upload_to_huggingface.py

필요한 패키지:
    pip install huggingface_hub
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("❌ huggingface_hub가 설치되지 않았습니다.")
    print("다음 명령어로 설치하세요: pip install huggingface_hub")
    sys.exit(1)


def upload_model(
    model_path: str,
    repo_id: str = "RomEscape/snow_white_gguf",
    repo_type: str = "model",
):
    """
    HuggingFace Hub에 모델 파일 업로드

    Args:
        model_path: 업로드할 모델 파일 경로
        repo_id: HuggingFace 저장소 ID (사용자명/저장소명)
        repo_type: 저장소 타입 ('model', 'dataset', 'space')
    """
    model_file = Path(model_path)
    
    if not model_file.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    file_size_gb = model_file.stat().st_size / (1024 ** 3)
    print(f"📦 모델 파일: {model_file.name}")
    print(f"📊 파일 크기: {file_size_gb:.2f} GB")
    print(f"🔗 저장소: {repo_id}")
    print()
    
    # HuggingFace 로그인 확인
    try:
        api = HfApi()
        whoami = api.whoami()
        print(f"✅ 로그인됨: {whoami['name']}")
    except Exception as e:
        print("❌ HuggingFace에 로그인되지 않았습니다.")
        print("다음 명령어로 로그인하세요: huggingface-cli login")
        print(f"오류: {e}")
        sys.exit(1)
    
    # 저장소가 없으면 생성 시도
    print(f"📦 저장소 확인 중...")
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"✅ 저장소가 이미 존재합니다")
    except Exception:
        print(f"📝 저장소가 없어서 생성 중...")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                exist_ok=True,
            )
            print(f"✅ 저장소 생성 완료")
        except Exception as e:
            print(f"⚠️  저장소 생성 실패 (웹에서 먼저 생성하거나 토큰 권한 확인): {e}")
            print(f"💡 해결 방법: https://huggingface.co/new 에서 저장소를 먼저 생성하세요")
            sys.exit(1)
    
    # 모델 업로드
    print(f"\n🚀 업로드 시작... (시간이 걸릴 수 있습니다)")
    try:
        api.upload_file(
            path_or_fileobj=str(model_file),
            path_in_repo=model_file.name,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"\n✅ 업로드 완료!")
        print(f"📥 다운로드 링크: https://huggingface.co/{repo_id}/blob/main/{model_file.name}")
        print(f"\n사용자는 다음 명령어로 다운로드할 수 있습니다:")
        print(f"  huggingface-cli download {repo_id} {model_file.name} --local-dir fairy_tale/models/snow_white_gguf/")
    except Exception as e:
        print(f"\n❌ 업로드 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 기본 모델 경로 (백업 폴더의 실제 파일 우선)
    backup_model = Path(__file__).parent.parent / "models" / "snow_white_gguf_backup_1214" / "model-q4_0.gguf"
    default_model = Path(__file__).parent.parent / "models" / "snow_white_gguf" / "model-q4_0.gguf"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    elif backup_model.exists() and backup_model.stat().st_size > 1024 * 1024:  # 1MB 이상
        model_path = str(backup_model)
        print(f"✅ 백업 폴더의 실제 모델 파일을 사용합니다: {model_path}")
    else:
        model_path = str(default_model)
        print(f"⚠️  기본 경로를 사용합니다: {model_path}")
    
    # 저장소 ID는 환경변수 또는 사용자 계정명 사용
    # HuggingFace API에서 현재 사용자 정보 가져오기
    try:
        api = HfApi()
        whoami = api.whoami()
        username = whoami.get("name", "PJiNH")
    except:
        username = os.getenv("HF_USERNAME", "PJiNH")
    
    default_repo_id = f"{username}/snow_white_gguf"
    repo_id = os.getenv("HF_REPO_ID", default_repo_id)
    
    print(f"📦 저장소: {repo_id}\n")
    
    upload_model(model_path, repo_id=repo_id)

