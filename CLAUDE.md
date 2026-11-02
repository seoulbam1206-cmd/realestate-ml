# 작업 규칙 (realestate-ml)

- 보고·답변은 한국어로
- **push · 가입 · 배포는 하지 않음.** 커밋까지만 (배포 버튼은 사용자가 직접)
- **파일 삭제**: 지울 파일 목록을 먼저 보여 주고 승인 받은 뒤에만
- **점수 설정은 바꾸지 않음**: `config/score.json` (가중치 s1, `rule_version` 2), 시세 모델 `config/lgbm.json` (c05),
  등급 이름 4개(매매 추천 · 매매 고려 · 매매 고려 대상 아님 · 매매 절대 불가)와 "데이터 부족" 표시.
  13장의 조정 1회는 이미 사용함 → 평가·사후 분석 결과를 보고 설정·경계·규칙을 고치지 않음 (다음 버전에서 따로)
- **TiDB 업로드**: 올릴 표 이름 · 열 · 행 수를 먼저 보여 주고 승인 받은 뒤에만
  (`uv run python scripts/publish.py --tables mvp3 --dry-run` 으로 확인)
- 공개 표에는 개별 거래 행을 올리지 않음 (집계만, 5건 미만이면 가격 비움)
- 화면은 Streamlit (`app/`), 로컬 확인: `uv run streamlit run app/Home.py`
