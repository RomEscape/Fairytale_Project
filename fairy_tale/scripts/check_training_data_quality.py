#!/usr/bin/env python3
"""
학습 데이터 품질 확인 스크립트
백설공주 언급 비율, 잘못된 페르소나 응답 등을 확인
"""

import json
import sys
from pathlib import Path
from loguru import logger

def analyze_training_data(jsonl_path: str):
    """학습 데이터 품질 분석"""
    total = 0
    has_sw_mention = 0
    has_dwarf = 0
    has_prince = 0
    has_intro = 0
    has_wrong = 0
    
    wrong_samples = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                total += 1
                output = data.get('output', '')
                
                if '백설공주 말하기:' in output:
                    sw_part = output.split('백설공주 말하기:')[1].split('<|eot|>')[0].strip()
                    
                    # 백설공주 언급 확인
                    if '백설공주' in sw_part or '백설' in sw_part:
                        has_sw_mention += 1
                    
                    # 페르소나 관련 키워드
                    if '난쟁이' in sw_part:
                        has_dwarf += 1
                    if '왕자' in sw_part:
                        has_prince += 1
                    
                    # 자기소개
                    if any(word in sw_part for word in ['저는', '나는', '제가', '안녕']):
                        has_intro += 1
                    
                    # 잘못된 페르소나
                    wrong_keywords = ['수호자', '새', '백조', '새야', '새예요', '새가', '새는']
                    if any(keyword in sw_part for keyword in wrong_keywords):
                        has_wrong += 1
                        if len(wrong_samples) < 10:  # 처음 10개만 저장
                            wrong_samples.append({
                                'line': line_num,
                                'response': sw_part[:200]
                            })
            except Exception as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue
    
    # 결과 출력
    print("=" * 80)
    print("학습 데이터 품질 분석 결과")
    print("=" * 80)
    print(f"\n총 샘플 수: {total:,}개")
    print(f"\n백설공주 언급:")
    print(f"  - 언급 샘플: {has_sw_mention:,}개 ({has_sw_mention/total*100:.1f}%)")
    print(f"  - 목표: 50% 이상")
    print(f"  - 상태: {'목표 달성' if has_sw_mention/total >= 0.5 else '목표 미달성'}")
    
    print(f"\n페르소나 관련 키워드:")
    print(f"  - 난쟁이 언급: {has_dwarf:,}개 ({has_dwarf/total*100:.1f}%)")
    print(f"  - 왕자 언급: {has_prince:,}개 ({has_prince/total*100:.1f}%)")
    print(f"  - 자기소개: {has_intro:,}개 ({has_intro/total*100:.1f}%)")
    
    print(f"\n품질 문제:")
    print(f"  - 잘못된 페르소나: {has_wrong:,}개 ({has_wrong/total*100:.1f}%)")
    if has_wrong > 0:
        print(f"  - 상태: 문제 발견")
        print(f"\n  잘못된 페르소나 샘플 (처음 5개):")
        for i, sample in enumerate(wrong_samples[:5], 1):
            print(f"    {i}. 라인 {sample['line']}: {sample['response']}")
    else:
        print(f"  - 상태: 문제 없음")
    
    # 최종 평가
    print("\n" + "=" * 80)
    print("최종 평가")
    print("=" * 80)
    
    sw_ratio = has_sw_mention / total
    wrong_ratio = has_wrong / total
    
    if sw_ratio >= 0.5 and wrong_ratio < 0.1:
        print("학습 데이터 품질: 우수")
        print("   - 백설공주 언급 비율이 목표 달성")
        print("   - 잘못된 페르소나 응답이 적음")
        print("   - 재학습 진행 가능")
    elif sw_ratio >= 0.3:
        print("학습 데이터 품질: 보통")
        print("   - 백설공주 언급 비율이 목표에 근접")
        print("   - 추가 데이터 생성 권장")
    else:
        print("학습 데이터 품질: 불량")
        print("   - 백설공주 언급 비율이 너무 낮음")
        print("   - 대화 데이터 재생성 필요")
    
    return {
        'total': total,
        'has_sw_mention': has_sw_mention,
        'sw_ratio': sw_ratio,
        'has_wrong': has_wrong,
        'wrong_ratio': wrong_ratio,
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_training_data_quality.py <jsonl_path>")
        print("Example: python check_training_data_quality.py processed/2025-12-19/prompted/prompted_agent_dialogue_snow_white.jsonl")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    
    if not Path(jsonl_path).exists():
        logger.error(f"File not found: {jsonl_path}")
        sys.exit(1)
    
    result = analyze_training_data(jsonl_path)
    
    # JSON 형식으로도 출력 (자동화용)
    import json as json_module
    print("\n" + "=" * 80)
    print("JSON 결과 (자동화용)")
    print("=" * 80)
    print(json_module.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

