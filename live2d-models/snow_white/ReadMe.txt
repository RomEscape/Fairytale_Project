Snow White Live2D Model
백설공주 Live2D 모델

이 폴더에는 백설공주 캐릭터의 Live2D 모델 파일들이 들어갑니다.

필요한 파일 구조:
snow_white/
  └── runtime/
      ├── snow_white.model3.json  (필수) - 모델 메인 설정 파일
      ├── snow_white.moc3          (필수) - 모델 데이터 파일
      ├── snow_white.physics3.json (선택) - 물리 시뮬레이션 설정
      ├── snow_white.pose3.json    (선택) - 포즈 설정
      ├── snow_white.cdi3.json     (선택) - 디스플레이 정보
      ├── textures/                (필수) - 텍스처 이미지 폴더
      │   └── texture_00.png 등
      ├── expressions/             (선택) - 표정 파일들
      │   └── exp_01.exp3.json 등
      └── motions/                 (선택) - 모션 파일들
          └── Idle/                - Idle 모션 그룹
              └── mtn_01.motion3.json 등

모델 파일 준비 방법:
1. Artlist.io 텍스트-이미지 AI로 백설공주 이미지 생성
2. Live2D Cubism Editor를 사용하여 모델 제작
3. 모델을 내보내서 위 구조에 맞게 배치

주의사항:
- snow_white.model3.json 파일의 경로가 올바른지 확인하세요
- 텍스처 파일 경로가 model3.json과 일치하는지 확인하세요
- emotionMap의 인덱스가 expressions 폴더의 파일 순서와 일치해야 합니다

