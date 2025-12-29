"""
데이터 생성 통합 스크립트

프로파일과 프롬프트를 기반으로 장면 데이터 생성 → 대화 데이터 생성 → FT 데이터 준비
Ollama를 사용하여 LLM 호출을 수행합니다.
참고 프로젝트의 run_open_llm_gen_data.py 구조를 따릅니다.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from loguru import logger

# scripts 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from data_generator import SceneDataGenerator, DialogueDataGenerator


def run_command(command: list[str], cwd: str | None = None) -> bool:
    """
    명령어 실행
    
    Args:
        command: 실행할 명령어 리스트
        cwd: 작업 디렉토리
    
    Returns:
        성공 여부
    """
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        logger.info(f"Command succeeded: {' '.join(command)}")
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(e.stderr)
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="동화 캐릭터 데이터 생성 파이프라인")
    parser.add_argument(
        "--character",
        type=str,
        required=True,
        choices=["snow_white", "little_mermaid", "peter_pan", "three_pigs_little"],
        help="캐릭터 이름"
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        choices=["gen_scene", "gen_dialogue"],
        help="프롬프트 이름 (gen_scene 또는 gen_dialogue)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="exaone3.5:2.4b",
        help="사용할 Ollama 모델명 (기본값: exaone3.5:2.4b - 한국어 특화 모델)"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama 서버 URL"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="생성 온도"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="korean",
        choices=["korean", "english"],
        help="언어"
    )
    parser.add_argument(
        "--skip-scene",
        action="store_true",
        help="장면 생성 건너뛰기"
    )
    parser.add_argument(
        "--skip-dialogue",
        action="store_true",
        help="대화 생성 건너뛰기"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="대화 생성 시 사용할 처리된 장면 데이터 파일 경로 (예: processed/2025-12-05/generated_agent_scene_snow_white-korean.json). 지정하지 않으면 자동으로 가장 최근 파일을 찾습니다."
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    now = datetime.now().strftime('%Y-%m-%d')
    
    # 1단계: 장면 데이터 생성
    if not args.skip_scene and (not args.prompt_name or args.prompt_name == "gen_scene"):
        logger.info("Step 1: Generating scenes with Ollama...")
        
        try:
            scene_generator = SceneDataGenerator(
                character=args.character,
                model=args.model_name,
                base_url=args.ollama_url,
                temperature=args.temperature,
            )
            
            scene_file = base_dir / "result" / now / "gen_scene" / f"scene_{args.model_name}-temp-{args.temperature}-char-{args.character}-{args.language}.jsonl"
            scene_file = scene_generator.generate_scenes(scene_file)
            logger.success(f"Scene generation completed: {scene_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate scenes: {e}")
            return 1
    
    # 2단계: 대화 데이터 생성
    if not args.skip_dialogue and (not args.prompt_name or args.prompt_name == "gen_dialogue"):
        logger.info("Step 2: Generating dialogues with Ollama...")
        
        try:
            dialogue_generator = DialogueDataGenerator(
                character=args.character,
                model=args.model_name,
                base_url=args.ollama_url,
                temperature=args.temperature,
            )
            
            # 기존 파일이 있는지 확인 (resume을 위해)
            dialogue_file_name = f"dialogue_{args.model_name}-temp-{args.temperature}-char-{args.character}-{args.language}.jsonl"
            result_dir = base_dir / "result"
            
            # 장면 데이터 파일 경로에서 날짜 추출 (우선순위 1)
            preferred_date = None
            if args.data_path:
                import re
                date_pattern = r'(\d{4}-\d{2}-\d{2})'
                match = re.search(date_pattern, str(args.data_path))
                if match:
                    preferred_date = match.group(1)
                    logger.debug(f"Extracted date from data_path: {preferred_date}")
            
            existing_dialogue_file = None
            if result_dir.exists():
                # 우선순위 1: 장면 데이터 파일과 같은 날짜의 대화 파일 찾기
                if preferred_date:
                    preferred_dir = result_dir / preferred_date
                    preferred_candidate = preferred_dir / "gen_dialogue" / dialogue_file_name
                    if preferred_candidate.exists():
                        existing_dialogue_file = preferred_candidate
                        logger.info(f"Found dialogue file matching scene data date: {existing_dialogue_file}")
                
                # 우선순위 2: 같은 날짜 파일이 없으면 가장 최근 날짜의 파일 찾기
                if not existing_dialogue_file:
                    date_dirs = sorted([d for d in result_dir.iterdir() if d.is_dir()], reverse=True)
                    for date_dir in date_dirs:
                        dialogue_file_candidate = date_dir / "gen_dialogue" / dialogue_file_name
                        if dialogue_file_candidate.exists():
                            existing_dialogue_file = dialogue_file_candidate
                            logger.info(f"Found existing dialogue file: {existing_dialogue_file}")
                            break
            
            # 항상 오늘 날짜로 새로 생성 (기존 파일과 무관하게)
            # 기존 파일이 있어도 새로운 프롬프트로 재생성할 수 있도록 오늘 날짜 사용
            dialogue_file = base_dir / "result" / now / "gen_dialogue" / dialogue_file_name
            if existing_dialogue_file and existing_dialogue_file != dialogue_file:
                logger.info(f"Note: Existing dialogue file found at {existing_dialogue_file}, but creating new file at {dialogue_file} with today's date")
            
            # 장면 데이터 경로 지정 (trainable-agents 방식)
            scene_data_path = None
            if args.data_path:
                input_path = Path(args.data_path)
                found_path = None
                
                if input_path.is_absolute():
                    # 절대 경로인 경우
                    if input_path.exists():
                        found_path = input_path.resolve()
                    else:
                        logger.error(f"Specified scene data file not found: {input_path}")
                        logger.info(f"Please check the file path and try again.")
                        return 1
                else:
                    # 상대 경로인 경우 여러 가능한 경로 시도
                    possible_paths = [
                        base_dir / args.data_path,
                        Path.cwd() / args.data_path,
                        Path(args.data_path),
                    ]
                    
                    for path in possible_paths:
                        if path.exists():
                            found_path = path.resolve()
                            break
                    
                    # 파일을 찾지 못한 경우, 디렉토리에서 비슷한 파일명 찾기 시도
                    if found_path is None:
                        path_obj = Path(args.data_path)
                        # 디렉토리 경로 추출 시도
                        dir_candidates = [
                            base_dir / path_obj.parent,
                            Path.cwd() / path_obj.parent,
                            path_obj.parent,
                        ]
                        
                        matching_files = []
                        searched_dir = None
                        
                        for dir_candidate in dir_candidates:
                            if dir_candidate.exists() and dir_candidate.is_dir():
                                # 캐릭터 이름이 포함된 scene 파일 찾기
                                pattern = f"generated_agent_scene_*{args.character}*.json"
                                matching_files = list(dir_candidate.glob(pattern))
                                if matching_files:
                                    searched_dir = dir_candidate
                                    break
                        
                        if matching_files:
                            # 가장 최근 파일 선택
                            found_path = max(matching_files, key=lambda p: p.stat().st_mtime).resolve()
                            logger.warning(
                                f"Exact file not found: {args.data_path}"
                            )
                            logger.info(
                                f"Found similar file in {searched_dir}: {found_path.name}. Using it instead."
                            )
                        else:
                            logger.error(f"Scene data file not found: {args.data_path}")
                            logger.info(f"Tried paths:")
                            for path in possible_paths:
                                logger.info(f"  - {path} (exists: {path.exists()})")
                            
                            # 실제 존재하는 파일 목록 표시
                            if searched_dir:
                                existing_files = list(searched_dir.glob(f"*{args.character}*.json"))
                                if existing_files:
                                    logger.info(f"\nAvailable files in {searched_dir}:")
                                    for f in existing_files:
                                        logger.info(f"  - {f.name}")
                            
                            return 1
                
                if found_path and found_path.exists():
                    scene_data_path = found_path
                    logger.info(f"Using specified scene data: {scene_data_path}")
                else:
                    logger.error(f"Scene data file not found: {args.data_path}")
                    return 1
            
            dialogue_file = dialogue_generator.generate_dialogues(
                dialogue_file, 
                scene_data_path=scene_data_path,
                resume=True  # 기존 파일이 있으면 중단 지점부터 이어서 생성
            )
            logger.success(f"Dialogue generation completed: {dialogue_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate dialogues: {e}")
            return 1
    
    logger.success("Data generation completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

