# Shizuku Live2D 모델 구조 및 움직임 원리 분석

## 개요
이 문서는 `shizuku` Live2D 모델의 내부 구조와 각 요소가 어떻게 작동하는지 상세히 설명합니다.

---

## 1. 전체 파일 구조

### 1.1 핵심 파일들

```
shizuku/
└── runtime/
    ├── shizuku.model3.json      # 모델 메타데이터 및 설정
    ├── shizuku.moc3              # 모델 데이터 (바이너리)
    ├── shizuku.cdi3.json         # 파라미터 및 파츠 정의
    ├── shizuku.physics3.json     # 물리 효과 설정
    ├── shizuku.pose3.json        # 포즈 그룹 설정
    ├── shizuku.1024/             # 텍스처 이미지 폴더
    │   ├── texture_00.png
    │   ├── texture_01.png
    │   ├── texture_02.png
    │   ├── texture_03.png
    │   └── texture_04.png
    └── motion/                   # 모션 파일들
        ├── 01.motion3.json       # FlickUp 모션
        ├── 02.motion3.json       # Tap 모션
        ├── 03.motion3.json       # Flick3 모션
        └── 04.motion3.json       # Idle 모션
```

---

## 2. 모델 구조 (model3.json)

### 2.1 파일 참조 구조

```json
{
  "FileReferences": {
    "Moc": "shizuku.moc3",              // 모델 데이터
    "Textures": [                        // 텍스처 이미지들
      "shizuku.1024/texture_00.png",
      "shizuku.1024/texture_01.png",
      "shizuku.1024/texture_02.png",
      "shizuku.1024/texture_03.png",
      "shizuku.1024/texture_04.png"
    ],
    "Physics": "shizuku.physics3.json",  // 물리 효과
    "Pose": "shizuku.pose3.json",        // 포즈 설정
    "DisplayInfo": "shizuku.cdi3.json",  // 파라미터/파츠 정보
    "Motions": {                         // 모션 그룹
      "FlickUp": [...],
      "Tap": [...],
      "Flick3": [...],
      "Idle": [...]
    }
  }
}
```

**역할:**
- 모든 파일의 경로와 관계를 정의
- 모델이 어떤 텍스처, 모션, 물리 효과를 사용하는지 명시

### 2.2 파라미터 그룹 (Groups)

```json
"Groups": [
  {
    "Target": "Parameter",
    "Name": "EyeBlink",
    "Ids": [
      "PARAM_EYE_L_OPEN",
      "PARAM_EYE_R_OPEN"
    ]
  },
  {
    "Target": "Parameter",
    "Name": "LipSync",
    "Ids": [
      "PARAM_MOUTH_OPEN_Y"
    ]
  }
]
```

**역할:**
- **EyeBlink 그룹**: 양쪽 눈 깜빡임을 하나의 그룹으로 묶어서 제어
- **LipSync 그룹**: 입 벌림 정도를 음성 동기화에 사용

**작동 원리:**
- 외부에서 `EyeBlink` 그룹에 값을 설정하면 양쪽 눈이 동시에 깜빡임
- `LipSync` 그룹은 음성 분석 결과에 따라 입을 움직임

---

## 3. 파라미터 시스템 (cdi3.json)

### 3.1 파라미터의 역할

파라미터는 Live2D 모델의 **애니메이션 제어 핵심**입니다. 각 파라미터는 0.0 ~ 1.0 (또는 -1.0 ~ 1.0) 범위의 값을 가집니다.

### 3.2 주요 파라미터 카테고리

#### 3.2.1 얼굴 회전 파라미터

```json
"PARAM_ANGLE_X"  // 얼굴 좌우 회전 (-30도 ~ +30도)
"PARAM_ANGLE_Y"  // 얼굴 상하 회전 (-30도 ~ +30도)
"PARAM_ANGLE_Z"  // 얼굴 좌우 기울임 (-30도 ~ +30도)
```

**작동 원리:**
- 사용자가 마우스로 캐릭터를 움직이면 이 값들이 변경됨
- 값에 따라 얼굴이 3D처럼 회전하는 효과 생성
- 예: `PARAM_ANGLE_X = 0.5` → 얼굴이 오른쪽으로 회전

#### 3.2.2 눈 파라미터

```json
"PARAM_EYE_L_OPEN"        // 왼쪽 눈 열림 (0.0: 닫힘, 1.0: 완전히 열림)
"PARAM_EYE_R_OPEN"        // 오른쪽 눈 열림
"PARAM_EYE_BALL_X"        // 눈동자 좌우 이동 (-1.0 ~ 1.0)
"PARAM_EYE_BALL_Y"        // 눈동자 상하 이동 (-1.0 ~ 1.0)
"PARAM_EYE_BALL_FORM"     // 눈동자 변형
"PARAM_EYE_BALL_KIRAKIRA" // 눈 반짝임 효과
```

**작동 원리:**
- `PARAM_EYE_L_OPEN`과 `PARAM_EYE_R_OPEN`을 0.0으로 설정하면 눈이 감김
- 자동 깜빡임: 주기적으로 0.0 → 1.0 → 0.0으로 변화
- 눈동자 추적: 마우스 위치에 따라 `PARAM_EYE_BALL_X/Y` 값 변경

#### 3.2.3 눈썹 파라미터

```json
"PARAM_BROW_L_Y"      // 왼쪽 눈썹 상하 이동
"PARAM_BROW_R_Y"      // 오른쪽 눈썹 상하 이동
"PARAM_BROW_L_X"      // 왼쪽 눈썹 좌우 이동
"PARAM_BROW_R_X"      // 오른쪽 눈썹 좌우 이동
"PARAM_BROW_L_ANGLE"  // 왼쪽 눈썹 각도
"PARAM_BROW_R_ANGLE"  // 오른쪽 눈썹 각도
"PARAM_BROW_L_FORM"   // 왼쪽 눈썹 변형
"PARAM_BROW_R_FORM"   // 오른쪽 눈썹 변형
```

**작동 원리:**
- 표정 표현에 사용: 슬픔(눈썹 올림), 화남(눈썹 내림) 등
- 각 파라미터가 독립적으로 작동하여 다양한 표정 생성

#### 3.2.4 입 파라미터

```json
"PARAM_MOUTH_OPEN_Y"  // 입 벌림 정도 (0.0: 닫힘, 1.0: 최대 벌림) ⭐ LipSync 핵심
"PARAM_MOUTH_FORM"    // 입 모양 변형 (웃음, 찡그림 등)
"PARAM_MOUTH_SIZE"    // 입 크기 조절
```

**작동 원리:**
- **LipSync**: 음성 분석 결과에 따라 `PARAM_MOUTH_OPEN_Y` 값이 실시간으로 변경
  - 예: "아" 소리 → 값 증가, "음" 소리 → 값 감소
- `PARAM_MOUTH_FORM`으로 다양한 입 모양 표현 (웃음, 찡그림 등)

#### 3.2.5 몸체 파라미터

```json
"PARAM_BODY_X"    // 몸통 X축 회전
"PARAM_BODY_Y"    // 몸통 Y축 회전
"PARAM_BODY_Z"    // 몸통 Z축 회전
"PARAM_BREATH"    // 호흡 애니메이션 (자동으로 주기적 변화)
```

**작동 원리:**
- `PARAM_BREATH`: 사인파 형태로 자동 변화하여 자연스러운 호흡 효과
- 몸통 회전: 얼굴과 함께 몸이 회전하는 효과

#### 3.2.6 팔/손 파라미터

```json
"PARAM_ARM_L"      // 왼팔 기본 회전
"PARAM_ARM_L_02"   // 왼팔 추가 회전 (팔꿈치)
"PARAM_HAND_L"     // 왼손 회전
"PARAM_ARM_R"      // 오른팔 기본 회전
"PARAM_ARM_R_02"   // 오른팔 추가 회전 (팔꿈치)
"PARAM_HAND_R"     // 오른손 회전
```

**작동 원리:**
- 팔은 상/하단으로 분리되어 있어 자연스러운 관절 움직임 가능
- 모션 파일에서 여러 파라미터를 동시에 제어하여 손동작 표현

#### 3.2.7 머리카락 물리 효과 파라미터

```json
"PARAM_KAMIYURE_FRONT"   // 앞머리 흔들림
"PARAM_KAMIYURE_BACK"    // 뒤머리 흔들림
"PARAM_KAMIYURE_SIDE_L"  // 왼쪽 옆머리 흔들림
"PARAM_KAMIYURE_SIDE_R"  // 오른쪽 옆머리 흔들림
"PARAM_KAMIYURE_TWIN_L"  // 왼쪽 트윈테일 흔들림
"PARAM_KAMIYURE_TWIN_R"  // 오른쪽 트윈테일 흔들림
```

**작동 원리:**
- 물리 효과 시스템에 의해 자동으로 계산됨 (직접 제어 불가)
- 얼굴 회전에 따라 자연스럽게 흔들림

---

## 4. 파츠 시스템 (Parts)

### 4.1 파츠의 역할

파츠는 모델을 구성하는 **시각적 요소**입니다. 각 파츠는 독립적으로 표시/숨김 처리할 수 있습니다.

### 4.2 주요 파츠 구조

#### 4.2.1 얼굴 파츠

```
PARTS_01_FACE_001          // 얼굴 기본형
PARTS_01_EYE_001          // 눈 기본형
PARTS_01_EYE_BALL_001     // 눈동자
PARTS_01_EYE_BALL_002     // 눈동자 (회전 효과용)
PARTS_01_EYE_BALL_003     // 눈동자 (반짝임 효과용)
PARTS_01_BROW_001         // 눈썹
PARTS_01_NOSE_001         // 코
PARTS_01_MOUTH_001        // 입
PARTS_01_HOHO_01          // 볼 (홍조 효과용)
PARTS_01_FACESHADOW_001   // 얼굴 그림자
```

**작동 원리:**
- 각 파츠는 독립적인 레이어로 렌더링됨
- 파츠 표시/숨김으로 다양한 표정 표현 가능
- 예: `PARTS_01_EYE_BALL_003`을 표시하면 눈이 반짝임

#### 4.2.2 머리카락 파츠

```
PARTS_01_HAIR_FRONT_001   // 앞머리
PARTS_01_HAIR_SIDE_001    // 옆머리
PARTS_01_HAIR_BACK_001    // 뒤머리 (트윈테일)
PARTS_01_HAIR_BACK_002    // 뒤머리 (내려뜬 머리)
PARTS_01_HAIR_TWIN_L      // 왼쪽 트윈테일
PARTS_01_HAIR_TWIN_R      // 오른쪽 트윈테일
```

**작동 원리:**
- 머리 스타일 변경: `PARTS_01_HAIR_BACK_001` 숨김 + `PARTS_01_HAIR_BACK_002` 표시
- 각 파츠는 물리 효과 파라미터와 연결되어 자연스럽게 흔들림

#### 4.2.3 몸체 파츠

```
PARTS_01_BODY             // 몸통
PARTS_01_NECK             // 목
PARTS_01_UNIFORM          //制服 (교복)
PARTS_01_PAJAMA           // パジャマ (잠옷)
PARTS_01_ARM_L_01         // 왼팔 상단
PARTS_01_ARM_L_02         // 왼팔 하단
PARTS_01_ARM_R_01         // 오른팔 상단
PARTS_01_ARM_R_02         // 오른팔 하단
PARTS_01_HAND_L           // 왼손
PARTS_01_HAND_R           // 오른손
```

**작동 원리:**
- 의상 변경: `PARTS_01_UNIFORM` 숨김 + `PARTS_01_PAJAMA` 표시
- 팔은 상/하단으로 분리되어 있어 관절처럼 움직임

---

## 5. 물리 효과 시스템 (physics3.json)

### 5.1 물리 효과의 역할

물리 효과는 **자동으로 계산되는 자연스러운 움직임**을 제공합니다. 특히 머리카락이 얼굴 회전에 따라 자연스럽게 흔들리는 효과를 만듭니다.

### 5.2 물리 효과 구조

#### 5.2.1 앞머리 흔들림 (PhysicsSetting1)

```json
{
  "Id": "PhysicsSetting1",
  "Input": [
    {
      "Source": { "Target": "Parameter", "Id": "PARAM_ANGLE_X" },
      "Weight": 60,
      "Type": "X"
    },
    {
      "Source": { "Target": "Parameter", "Id": "PARAM_ANGLE_Z" },
      "Weight": 60,
      "Type": "Angle"
    }
  ],
  "Output": [
    {
      "Destination": { "Target": "Parameter", "Id": "PARAM_KAMIYURE_FRONT" },
      "VertexIndex": 1,
      "Scale": 2,
      "Weight": 100
    }
  ],
  "Vertices": [
    { "Position": { "X": 0, "Y": 0 }, "Mobility": 1, "Delay": 1 },
    { "Position": { "X": 0, "Y": 3 }, "Mobility": 0.95, "Delay": 0.9 }
  ]
}
```

**작동 원리:**
1. **Input**: 얼굴 회전 각도 (`PARAM_ANGLE_X`, `PARAM_ANGLE_Z`)를 입력으로 받음
2. **물리 계산**: 두 개의 버텍스(점)로 구성된 진자 시스템으로 계산
   - 첫 번째 버텍스: 고정점 (Mobility: 1.0, Delay: 1.0)
   - 두 번째 버텍스: 움직이는 점 (Mobility: 0.95, Delay: 0.9)
3. **Output**: 계산 결과를 `PARAM_KAMIYURE_FRONT` 파라미터로 출력
4. **효과**: 앞머리가 얼굴 회전에 따라 자연스럽게 흔들림

**파라미터 설명:**
- **Mobility**: 움직임 민감도 (1.0 = 완전히 움직임, 0.0 = 고정)
- **Delay**: 반응 지연 시간 (값이 작을수록 빠르게 반응)
- **Acceleration**: 가속도 (값이 클수록 빠르게 움직임)
- **Radius**: 움직임 반경

#### 5.2.2 옆머리 흔들림 (PhysicsSetting2)

```json
{
  "Id": "PhysicsSetting2",
  "Output": [
    { "Destination": { "Id": "PARAM_KAMIYURE_SIDE_L" } },
    { "Destination": { "Id": "PARAM_KAMIYURE_SIDE_R" } }
  ]
}
```

**작동 원리:**
- 왼쪽/오른쪽 옆머리를 각각 제어
- 같은 입력(얼굴 회전)을 받지만 좌우 대칭으로 반대 방향으로 움직임

#### 5.2.3 트윈테일 흔들림 (PhysicsSetting3)

```json
{
  "Id": "PhysicsSetting3",
  "Output": [
    { "Destination": { "Id": "PARAM_KAMIYURE_TWIN_L" } },
    { "Destination": { "Id": "PARAM_KAMIYURE_TWIN_R" } }
  ],
  "Vertices": [
    { "Position": { "X": 0, "Y": 0 } },
    { "Position": { "X": 0, "Y": 10 } }  // 더 긴 진자
  ]
}
```

**작동 원리:**
- 트윈테일은 더 긴 진자(Y: 10)로 설정되어 더 크게 흔들림
- 좌우 대칭으로 움직임

### 5.3 중력 설정

```json
"EffectiveForces": {
  "Gravity": { "X": 0, "Y": -1 },  // 아래 방향 중력
  "Wind": { "X": 0, "Y": 0 }       // 바람 없음
}
```

**작동 원리:**
- 중력이 Y축 -1 방향으로 작용하여 머리카락이 아래로 떨어지는 효과
- 바람 효과를 추가하면 좌우로 흔들릴 수 있음

---

## 6. 포즈 시스템 (pose3.json)

### 6.1 포즈의 역할

포즈는 **여러 파츠를 그룹으로 묶어서 함께 제어**하는 시스템입니다.

### 6.2 포즈 구조

```json
{
  "Groups": [
    [
      { "Id": "PARTS_01_ARM_R_02" },  // 오른팔 하단
      { "Id": "PARTS_01_ARM_R_01" }   // 오른팔 상단
    ],
    [
      { "Id": "PARTS_01_ARM_L_02" },  // 왼팔 하단
      { "Id": "PARTS_01_ARM_L_01" }   // 왼팔 상단
    ]
  ]
}
```

**작동 원리:**
- 오른팔 상/하단이 하나의 그룹으로 묶여 있음
- 한쪽 팔의 파라미터를 변경하면 상/하단이 함께 움직임
- 자연스러운 관절 움직임을 보장

**예시:**
- `PARAM_ARM_R` 값을 변경하면 → 오른팔 상단이 움직임
- 오른팔 상단이 움직이면 → 포즈 시스템이 하단도 자동으로 조정

---

## 7. 모션 시스템 (motion3.json)

### 7.1 모션의 역할

모션은 **시간에 따라 파라미터 값이 변화하는 애니메이션**입니다.

### 7.2 모션 구조 예시 (FlickUp)

```json
{
  "Meta": {
    "Duration": 1.27,      // 모션 길이 (초)
    "Fps": 30.0,           // 초당 프레임 수
    "Loop": true           // 반복 재생
  },
  "Curves": [
    {
      "Target": "Parameter",
      "Id": "PARAM_ANGLE_X",
      "Segments": [
        0, 0,              // 시작: 시간 0, 값 0
        1, 0.1,            // 베지어 곡선
        0, 0.2,
        1, 0.3,
        1, 1,              // 시간 0.422, 값 1
        0.422, 1,
        0.544, -1,         // 시간 0.544, 값 -1
        0.667, -1,
        0, 1.267,          // 시간 1.267, 값 0 (원래대로)
        -1
      ]
    }
  ]
}
```

**작동 원리:**
1. **시간축**: 0초부터 1.27초까지
2. **파라미터 변화**: `PARAM_ANGLE_X` 값이 0 → 1 → -1 → 0으로 변화
3. **베지어 곡선**: 부드러운 애니메이션을 위한 곡선 보간
4. **결과**: 얼굴이 오른쪽 → 왼쪽 → 중앙으로 부드럽게 움직임

### 7.3 모션 타입

#### FlickUp (01.motion3.json)
- 위로 스와이프할 때 재생
- 얼굴이 위를 향하며 눈이 크게 뜸

#### Tap (02.motion3.json)
- 캐릭터를 탭할 때 재생
- 반응하는 애니메이션 (눈 깜빡임, 표정 변화)

#### Flick3 (03.motion3.json)
- 특정 방향 스와이프 시 재생
- 다양한 방향으로 움직임

#### Idle (04.motion3.json)
- 대기 상태에서 자동 재생
- 호흡, 미세한 움직임 등

---

## 8. 텍스처 시스템

### 8.1 텍스처 레이어 구조

```
texture_00.png  // 배경, 코어 파츠 (가장 뒤)
texture_01.png  // 캐릭터 기본 레이어
texture_02.png  // 캐릭터 상세 레이어
texture_03.png  // 팔, 손 등 추가 레이어
texture_04.png  // 눈, 입 등 표정 레이어 (가장 앞)
```

**작동 원리:**
- 각 텍스처는 독립적인 레이어로 렌더링됨
- 뒤에서 앞 순서로 합성되어 최종 이미지 생성
- 파츠별로 다른 텍스처에 있을 수 있음

### 8.2 텍스처 해상도

- **1024x1024 픽셀**: 모든 텍스처가 동일한 해상도
- Live2D는 텍스처를 메모리에 로드하여 실시간으로 변형

---

## 9. 전체 움직임 흐름

### 9.1 실시간 애니메이션 흐름

```
1. 사용자 입력 (마우스 이동)
   ↓
2. 파라미터 값 변경 (PARAM_ANGLE_X, PARAM_ANGLE_Y 등)
   ↓
3. 물리 효과 계산 (physics3.json)
   - 얼굴 회전 → 머리카락 흔들림 파라미터 자동 계산
   ↓
4. 포즈 시스템 적용 (pose3.json)
   - 팔 상/하단 함께 움직임
   ↓
5. 파츠 변형
   - 각 파라미터 값에 따라 파츠가 변형됨
   ↓
6. 텍스처 렌더링
   - 변형된 파츠를 텍스처 이미지로 렌더링
   ↓
7. 최종 화면 출력
```

### 9.2 모션 재생 흐름

```
1. 모션 트리거 (사용자 액션 또는 자동)
   ↓
2. 모션 파일 로드 (motion3.json)
   ↓
3. 시간에 따른 파라미터 값 계산
   - 베지어 곡선으로 부드러운 보간
   ↓
4. 파라미터 값 적용
   ↓
5. 물리 효과 + 포즈 시스템 적용
   ↓
6. 렌더링 및 화면 출력
```

---

## 10. 핵심 개념 정리

### 10.1 파라미터 vs 파츠

- **파라미터**: 숫자 값 (0.0 ~ 1.0)으로 움직임을 제어하는 **제어 장치**
- **파츠**: 실제로 보이는 **시각적 요소**

**관계:**
- 파라미터 값 변경 → 파츠가 변형됨
- 예: `PARAM_EYE_L_OPEN = 0.0` → 눈 파츠가 닫힘

### 10.2 물리 효과의 자동성

- 물리 효과 파라미터는 **직접 제어 불가**
- 다른 파라미터(얼굴 회전)에 의해 **자동으로 계산됨**
- 예: `PARAM_KAMIYURE_FRONT`는 `PARAM_ANGLE_X`에 의해 자동 계산

### 10.3 모션의 시간성

- 모션은 **시간에 따라 변화하는 파라미터 값의 시퀀스**
- 실시간 제어와 달리 **미리 정의된 애니메이션**

---

## 11. 실제 사용 예시

### 11.1 눈 깜빡임 구현

```javascript
// 자동 깜빡임 (주기적으로 실행)
setInterval(() => {
  model.setParameterValue("PARAM_EYE_L_OPEN", 0.0);  // 눈 감김
  model.setParameterValue("PARAM_EYE_R_OPEN", 0.0);
  
  setTimeout(() => {
    model.setParameterValue("PARAM_EYE_L_OPEN", 1.0);  // 눈 뜸
    model.setParameterValue("PARAM_EYE_R_OPEN", 1.0);
  }, 100);
}, 3000);
```

### 11.2 LipSync 구현

```javascript
// 음성 분석 결과에 따라 입 움직임
function updateLipSync(volume) {
  const mouthOpen = Math.min(volume * 2, 1.0);  // 볼륨을 0~1로 변환
  model.setParameterValue("PARAM_MOUTH_OPEN_Y", mouthOpen);
}
```

### 11.3 모션 재생

```javascript
// FlickUp 모션 재생
model.startMotion("FlickUp", 0);  // 그룹명, 우선순위
```

---

## 12. 요약

### Shizuku 모델의 핵심 구조:

1. **5개 텍스처 레이어**: 배경부터 표정까지 계층적 구성
2. **40개 이상의 파라미터**: 얼굴, 눈, 입, 몸체, 팔 등 모든 움직임 제어
3. **40개 이상의 파츠**: 각 부위를 독립적으로 제어 가능
4. **3개 물리 효과**: 머리카락 자연스러운 흔들림
5. **2개 포즈 그룹**: 팔 상/하단 연결
6. **4개 기본 모션**: 다양한 상호작용 애니메이션

### 움직임의 핵심:

- **파라미터 값 변경** → **파츠 변형** → **화면 출력**
- **물리 효과**: 자동 계산으로 자연스러운 움직임
- **모션**: 시간에 따른 파라미터 변화로 애니메이션

이 구조를 이해하면 Live2D 모델을 제작하고 커스터마이징할 수 있습니다!

