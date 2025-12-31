# Snow White 모델 구조 검증 및 Shizuku 비교 분석

## 검증 결과 요약

✅ **snow_white 모델은 shizuku 모델과 구조가 완전히 동일하며, 모든 필수 파일이 올바르게 구성되어 있습니다.**

---

## 1. 파일 구조 비교

### 1.1 핵심 파일 존재 여부

| 파일 | snow_white | shizuku | 상태 |
|------|------------|---------|------|
| `model3.json` | ✅ | ✅ | 동일 |
| `moc3` | ✅ | ✅ | 존재 |
| `cdi3.json` | ✅ | ✅ | 동일 |
| `physics3.json` | ✅ | ✅ | 동일 |
| `pose3.json` | ✅ | ✅ | 동일 |
| `texture_00.png` | ✅ | ✅ | 존재 |
| `texture_01.png` | ✅ | ✅ | 존재 |
| `texture_02.png` | ✅ | ✅ | 존재 |
| `texture_03.png` | ✅ | ✅ | 존재 |
| `texture_04.png` | ✅ | ✅ | 존재 |
| `motion/*.motion3.json` | ✅ | ✅ | 존재 |

**결론**: 모든 필수 파일이 존재하며 구조가 동일합니다.

---

## 2. model3.json 비교

### 2.1 텍스처 경로 비교

**snow_white:**
```json
"Textures": [
  "snow_white.1024/texture_00.png",
  "snow_white.1024/texture_01.png",
  "snow_white.1024/texture_02.png",
  "snow_white.1024/texture_03.png",
  "snow_white.1024/texture_04.png"
]
```

**shizuku:**
```json
"Textures": [
  "shizuku.1024/texture_00.png",
  "shizuku.1024/texture_01.png",
  "shizuku.1024/texture_02.png",
  "shizuku.1024/texture_03.png",
  "shizuku.1024/texture_04.png"
]
```

**검증 결과:**
- ✅ 경로 구조가 올바름 (`snow_white.1024/` 폴더)
- ✅ 파일 이름이 일치함 (`texture_00.png` ~ `texture_04.png`)
- ✅ 실제 파일이 존재함 (확인 완료)

### 2.2 모션 그룹 비교

**snow_white:**
```json
"Motions": {
  "FlickUp": [{"File": "motion/01.motion3.json"}],
  "Tap": [{"File": "motion/02.motion3.json"}],
  "Flick3": [{"File": "motion/03.motion3.json"}],
  "Idle": [{"File": "motion/04.motion3.json"}]
}
```

**shizuku:**
```json
"Motions": {
  "FlickUp": [{"File": "motion/01.motion3.json"}],
  "Tap": [{"File": "motion/02.motion3.json"}],
  "Flick3": [{"File": "motion/03.motion3.json"}],
  "Idle": [{"File": "motion/04.motion3.json"}]
}
```

**검증 결과:**
- ✅ 모션 그룹 구조가 완전히 동일함
- ✅ 모든 모션 파일 경로가 올바름

### 2.3 파라미터 그룹 비교

**snow_white:**
```json
"Groups": [
  {
    "Target": "Parameter",
    "Name": "EyeBlink",
    "Ids": ["PARAM_EYE_L_OPEN", "PARAM_EYE_R_OPEN"]
  },
  {
    "Target": "Parameter",
    "Name": "LipSync",
    "Ids": ["PARAM_MOUTH_OPEN_Y"]
  }
]
```

**shizuku:**
```json
"Groups": [
  {
    "Target": "Parameter",
    "Name": "EyeBlink",
    "Ids": ["PARAM_EYE_L_OPEN", "PARAM_EYE_R_OPEN"]
  },
  {
    "Target": "Parameter",
    "Name": "LipSync",
    "Ids": ["PARAM_MOUTH_OPEN_Y"]
  }
]
```

**검증 결과:**
- ✅ 파라미터 그룹이 완전히 동일함
- ✅ EyeBlink, LipSync 그룹이 올바르게 설정됨

---

## 3. cdi3.json 비교 (파라미터 및 파츠)

### 3.1 파라미터 비교

**총 파라미터 개수:**
- snow_white: **28개 파라미터**
- shizuku: **28개 파라미터**
- ✅ **완전히 동일**

**주요 파라미터 카테고리:**
- 얼굴 회전: `PARAM_ANGLE_X/Y/Z` ✅
- 눈: `PARAM_EYE_L_OPEN`, `PARAM_EYE_R_OPEN`, `PARAM_EYE_BALL_X/Y` 등 ✅
- 눈썹: `PARAM_BROW_L/R_Y/X/ANGLE/FORM` 등 ✅
- 입: `PARAM_MOUTH_OPEN_Y`, `PARAM_MOUTH_FORM`, `PARAM_MOUTH_SIZE` ✅
- 몸체: `PARAM_BODY_X/Y/Z`, `PARAM_BREATH` ✅
- 팔/손: `PARAM_ARM_L/R`, `PARAM_HAND_L/R` 등 ✅
- 머리카락 물리: `PARAM_KAMIYURE_FRONT/BACK/SIDE_L/R/TWIN_L/R` ✅

### 3.2 파츠 비교

**총 파츠 개수:**
- snow_white: **40개 파츠**
- shizuku: **40개 파츠**
- ✅ **완전히 동일**

**주요 파츠 카테고리:**
- 얼굴 파츠: `PARTS_01_FACE_001`, `PARTS_01_EYE_001` 등 ✅
- 머리카락 파츠: `PARTS_01_HAIR_FRONT_001`, `PARTS_01_HAIR_SIDE_001` 등 ✅
- 몸체 파츠: `PARTS_01_BODY`, `PARTS_01_NECK` 등 ✅
- 팔/손 파츠: `PARTS_01_ARM_L/R_01/02`, `PARTS_01_HAND_L/R` 등 ✅

**검증 결과:**
- ✅ 모든 파라미터 ID가 동일함
- ✅ 모든 파츠 ID가 동일함
- ✅ 구조가 완전히 호환됨

---

## 4. physics3.json 비교

### 4.1 물리 효과 설정 비교

**물리 효과 개수:**
- snow_white: **3개** (앞머리, 옆머리, 트윈테일)
- shizuku: **3개** (앞머리, 옆머리, 트윈테일)
- ✅ **완전히 동일**

**물리 효과 구조:**
```json
{
  "PhysicsSetting1": "前髪揺れ" (앞머리 흔들림),
  "PhysicsSetting2": "横髪揺れ" (옆머리 흔들림),
  "PhysicsSetting3": "ツインテ揺れ" (트윈테일 흔들림)
}
```

**검증 결과:**
- ✅ 물리 효과 설정이 동일함
- ✅ 입력 파라미터 (`PARAM_ANGLE_X`, `PARAM_ANGLE_Z`) 동일
- ✅ 출력 파라미터 (`PARAM_KAMIYURE_*`) 동일
- ✅ 중력 설정 동일 (`Y: -1`)

---

## 5. pose3.json 비교

### 5.1 포즈 그룹 비교

**snow_white:**
```json
"Groups": [
  [
    {"Id": "PARTS_01_ARM_R_02"},
    {"Id": "PARTS_01_ARM_R_01"}
  ],
  [
    {"Id": "PARTS_01_ARM_L_02"},
    {"Id": "PARTS_01_ARM_L_01"}
  ]
]
```

**shizuku:**
```json
"Groups": [
  [
    {"Id": "PARTS_01_ARM_R_02"},
    {"Id": "PARTS_01_ARM_R_01"}
  ],
  [
    {"Id": "PARTS_01_ARM_L_02"},
    {"Id": "PARTS_01_ARM_L_01"}
  ]
]
```

**검증 결과:**
- ✅ 포즈 그룹이 완전히 동일함
- ✅ 팔 상/하단 연결 구조가 동일함

---

## 6. PNG 파일 검증

### 6.1 텍스처 파일 존재 확인

**snow_white.1024/ 폴더:**
- ✅ `texture_00.png` - 존재
- ✅ `texture_01.png` - 존재
- ✅ `texture_02.png` - 존재
- ✅ `texture_03.png` - 존재
- ✅ `texture_04.png` - 존재

**model3.json에서 참조하는 경로:**
```json
"snow_white.1024/texture_00.png"
"snow_white.1024/texture_01.png"
"snow_white.1024/texture_02.png"
"snow_white.1024/texture_03.png"
"snow_white.1024/texture_04.png"
```

**검증 결과:**
- ✅ 모든 텍스처 파일이 올바른 경로에 존재함
- ✅ 파일 이름이 model3.json과 일치함
- ✅ 폴더 구조가 올바름 (`snow_white.1024/`)

### 6.2 PNG 파일 호환성 분석

**중요 사항:**
1. **파일 이름**: ✅ 올바름 (`texture_00.png` ~ `texture_04.png`)
2. **경로 구조**: ✅ 올바름 (`snow_white.1024/` 폴더)
3. **파일 개수**: ✅ 5개 모두 존재
4. **모델 참조**: ✅ model3.json에서 올바르게 참조됨

**잠재적 문제점:**
- ⚠️ PNG 파일의 **내부 구조** (파츠 배치, UV 좌표 등)는 `.moc3` 파일과 일치해야 함
- ⚠️ PNG 파일의 **해상도**가 1024x1024인지 확인 필요 (파일 크기로 추정 가능)
- ⚠️ PNG 파일의 **알파 채널** (투명도)이 올바른지 확인 필요

**확인 방법:**
- 실제로 모델을 로드해서 테스트해봐야 함
- Live2D Viewer나 애플리케이션에서 로드 시도

---

## 7. 전체 호환성 검증

### 7.1 구조적 호환성

| 항목 | snow_white | shizuku | 호환성 |
|------|------------|---------|--------|
| 모델 버전 | Version 3 | Version 3 | ✅ 동일 |
| 파라미터 구조 | 28개 | 28개 | ✅ 동일 |
| 파츠 구조 | 40개 | 40개 | ✅ 동일 |
| 물리 효과 | 3개 | 3개 | ✅ 동일 |
| 포즈 그룹 | 2개 | 2개 | ✅ 동일 |
| 모션 그룹 | 4개 | 4개 | ✅ 동일 |
| 텍스처 레이어 | 5개 | 5개 | ✅ 동일 |

### 7.2 파일 경로 호환성

**snow_white:**
- 모든 파일 경로가 `snow_white.*` 형식으로 일관성 있게 명명됨
- 폴더 구조가 shizuku와 동일함

**검증 결과:**
- ✅ 파일 명명 규칙이 일관됨
- ✅ 상대 경로가 올바름
- ✅ 모든 참조가 올바르게 설정됨

---

## 8. 작동 가능성 평가

### 8.1 ✅ 작동 가능한 이유

1. **구조적 호환성**: snow_white와 shizuku의 구조가 완전히 동일함
2. **파일 존재**: 모든 필수 파일이 올바른 위치에 존재함
3. **경로 일치**: model3.json에서 참조하는 경로와 실제 파일 경로가 일치함
4. **파라미터/파츠 일치**: cdi3.json의 파라미터와 파츠가 moc3 파일과 호환될 것으로 예상됨

### 8.2 ⚠️ 확인이 필요한 사항

1. **PNG 파일 내부 구조**:
   - PNG 파일의 파츠 배치가 `.moc3` 파일과 일치해야 함
   - UV 좌표 매핑이 올바른지 확인 필요
   - 실제 로드 테스트 필요

2. **PNG 파일 해상도**:
   - 모든 텍스처가 1024x1024 픽셀인지 확인 필요
   - 해상도가 다르면 렌더링 문제 발생 가능

3. **PNG 파일 알파 채널**:
   - 투명 배경이 올바르게 설정되어 있는지 확인 필요
   - 알파 채널 문제로 인한 렌더링 오류 가능

4. **moc3 파일 호환성**:
   - `.moc3` 파일이 새로운 PNG 파일과 호환되는지 확인 필요
   - 만약 PNG 파일을 교체했다면, `.moc3` 파일도 재생성해야 할 수 있음

---

## 9. 권장 사항

### 9.1 즉시 확인 가능한 사항

1. **파일 크기 확인**:
   ```bash
   ls -lh snow_white/runtime/snow_white.1024/*.png
   ```
   - 모든 파일이 비슷한 크기인지 확인
   - 너무 작거나 큰 파일이 있으면 문제 가능

2. **파일 형식 확인**:
   ```bash
   file snow_white/runtime/snow_white.1024/*.png
   ```
   - PNG 형식인지 확인
   - 손상된 파일이 있는지 확인

### 9.2 실제 테스트 필요

1. **Live2D Viewer에서 로드 테스트**:
   - `snow_white.model3.json` 파일을 Live2D Viewer에서 열기
   - 모델이 정상적으로 로드되는지 확인
   - 텍스처가 올바르게 표시되는지 확인

2. **애플리케이션에서 테스트**:
   - Open-LLM-VTuber 애플리케이션에서 모델 로드 시도
   - 파라미터 제어가 정상 작동하는지 확인
   - 애니메이션이 정상 작동하는지 확인

### 9.3 문제 발생 시 대응

**만약 모델이 로드되지 않는다면:**
1. PNG 파일을 shizuku의 PNG 파일로 교체해서 테스트
2. `.moc3` 파일이 새로운 PNG와 호환되는지 확인
3. Live2D Cubism Editor에서 모델을 다시 내보내기

**만약 텍스처가 잘못 표시된다면:**
1. PNG 파일의 파츠 배치가 `.moc3`와 일치하는지 확인
2. UV 좌표 매핑 확인
3. PNG 파일을 Live2D Cubism Editor에서 다시 임포트

---

## 10. 최종 결론

### ✅ 구조적 검증 결과

**snow_white 모델은 shizuku 모델과 구조가 완전히 동일하며, 모든 필수 파일이 올바르게 구성되어 있습니다.**

- ✅ 모든 JSON 설정 파일이 올바름
- ✅ 모든 텍스처 파일이 올바른 경로에 존재함
- ✅ 파라미터와 파츠 구조가 완전히 호환됨
- ✅ 물리 효과 및 포즈 설정이 올바름

### ⚠️ 추가 확인 필요

**PNG 파일의 실제 호환성은 다음을 확인해야 합니다:**

1. **실제 로드 테스트**: Live2D Viewer나 애플리케이션에서 모델 로드
2. **PNG 파일 내부 구조**: 파츠 배치와 UV 좌표가 `.moc3`와 일치하는지
3. **해상도 및 형식**: 모든 텍스처가 올바른 해상도와 형식인지

**예상 결과:**
- 구조적으로는 완벽하게 호환됨
- PNG 파일이 올바르게 제작되었다면 정상 작동할 것으로 예상됨
- 실제 테스트를 통해 최종 확인 필요

---

## 11. 체크리스트

### 구조 검증 ✅
- [x] model3.json 구조 확인
- [x] cdi3.json 파라미터/파츠 확인
- [x] physics3.json 물리 효과 확인
- [x] pose3.json 포즈 그룹 확인
- [x] 텍스처 파일 존재 확인
- [x] 모션 파일 존재 확인

### 실제 테스트 필요 ⚠️
- [ ] Live2D Viewer에서 모델 로드 테스트
- [ ] PNG 파일 해상도 확인 (1024x1024)
- [ ] PNG 파일 알파 채널 확인
- [ ] 애플리케이션에서 실제 작동 테스트
- [ ] 파라미터 제어 테스트
- [ ] 애니메이션 테스트

