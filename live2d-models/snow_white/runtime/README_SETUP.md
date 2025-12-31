# 백설공주 Live2D 모델 설정 가이드

## 📁 필요한 파일 구조

이 `runtime` 폴더에 다음 파일들이 필요합니다:

### 필수 파일:
- `snow_white.model3.json` - 모델 메인 설정 파일 (반드시 필요)
- `snow_white.moc3` - 모델 데이터 파일 (반드시 필요)
- `textures/` 폴더 - 텍스처 이미지들

### 선택 파일:
- `snow_white.physics3.json` - 물리 시뮬레이션 설정
- `snow_white.pose3.json` - 포즈 설정
- `snow_white.cdi3.json` - 디스플레이 정보
- `expressions/` 폴더 - 표정 파일들 (exp_01.exp3.json 등)
- `motions/` 폴더 - 모션 파일들

## 🎨 Artlist.io 텍스트-이미지 AI로 모델 만들기

### 1단계: 이미지 생성
1. https://artlist.io/text-to-image-ai 에서 백설공주 이미지 생성
2. 고해상도 이미지로 생성 (최소 1024x1024 권장)
3. 여러 각도/표정의 이미지 생성 (정면, 측면, 다양한 표정)

### 2단계: Live2D Cubism Editor로 변환
1. Live2D Cubism Editor 다운로드 및 설치
2. 새 프로젝트 생성
3. 생성한 이미지를 레이어로 가져오기
4. 파츠 분리 및 리깅 작업:
   - 얼굴 파츠 분리 (눈, 코, 입, 머리카락 등)
   - 신체 파츠 분리 (상체, 하체, 팔 등)
5. 파라미터 설정 (눈 깜빡임, 입 움직임 등)
6. 모션 및 표정 추가
7. 모델 내보내기:
   - File > Export > For Runtime
   - 모델명을 `snow_white`로 설정
   - 이 `runtime` 폴더에 내보내기

### 3단계: 파일 확인
내보낸 파일들이 다음 구조로 배치되었는지 확인:
```
runtime/
├── snow_white.model3.json
├── snow_white.moc3
├── snow_white.physics3.json (있으면)
├── snow_white.pose3.json (있으면)
├── snow_white.cdi3.json (있으면)
├── textures/
│   └── texture_00.png (또는 다른 이름)
├── expressions/ (있으면)
│   ├── exp_01.exp3.json
│   └── ...
└── motions/ (있으면)
    └── Idle/
        └── mtn_01.motion3.json
```

## ⚙️ model3.json 파일 확인

`snow_white.model3.json` 파일을 열어서 경로가 올바른지 확인하세요:
- 모든 파일 경로가 상대 경로로 설정되어 있어야 합니다
- 텍스처 파일 경로가 실제 파일 위치와 일치해야 합니다

## 🎭 표정 설정 (emotionMap)

표정 파일이 있다면, `model_dict.json`의 `emotionMap`이 표정 파일 순서와 일치하는지 확인:
- exp_01.exp3.json → neutral (0)
- exp_02.exp3.json → fear (1)
- exp_03.exp3.json → sadness (1)
- exp_04.exp3.json → anger (2)
- exp_05.exp3.json → disgust (2)
- exp_06.exp3.json → joy (3)
- exp_07.exp3.json → smirk (3)
- exp_08.exp3.json → surprise (3)

## ✅ 완료 확인

모든 파일이 준비되면:
1. 서버를 재시작하세요
2. 프론트엔드에서 캐릭터 선택 메뉴에서 "Snow White"를 선택할 수 있어야 합니다
3. 아바타 이미지(`snow_white.png`)를 `avatars/` 폴더에 추가하는 것을 잊지 마세요!

