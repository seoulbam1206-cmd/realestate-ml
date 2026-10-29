# 부동산 가격 흐름 · 저평가 탐지 · 온비드 공매 분석

> **공공 자료를 가공한 것이며 투자 조언이 아닙니다.** 개인 포트폴리오용 데모입니다.

전국 아파트 가격이 어디서 얼마나 내려왔는지 추적하고, 모델 시세 대비 저평가 지역·공매 물건을 찾는 ML 프로젝트입니다.
전체 계획은 [PLAN.md](PLAN.md), 데이터 확인 기록은 [DATA_CHECK.md](DATA_CHECK.md)에 있습니다.

## 진행 상황
| 단계 | 내용 | 상태 |
|---|---|---|
| MVP 1 | R-ONE 지수로 지역 가격 흐름 화면 | ✅ 로컬 완료, 공개 준비 |
| MVP 2 | 실거래 수집 파이프라인 (공개시스템 CSV + API) | 예정 |
| MVP 3 | 시세 모델, 점수, 백테스트 (LightGBM, MLflow) | 예정 |
| MVP 4 | 온비드 공매 연동, FastAPI | 예정 |
| MVP 5 | 스케줄, 재학습, 모니터링, Docker | 예정 |

## MVP 1 — 지역 가격 흐름
- 한국부동산원 R-ONE 월간 CSV 2개(CP949, 여러 줄 헤더, 가로형)를 세로형 Parquet로 정리
- 236개 지역을 `묶음 / 시도 / 권역 / 시군구 / 일반구` 계층으로 분류, 행정구역 개편(전남광주 통합, 인천·화성 구 개편) 반영
- 지표: 고점 대비 하락률, 고점 이후 바닥 대비 반등률, 3·6·12개월 변화율 (기준월 `asof` 를 받아 이후 백테스트에서 그대로 재사용)
- 화면: 홈(요약) · 지역 가격 흐름(지역 비교, 기준 다시 맞추기) · 하락 지역 순위 · 출처 안내

## 실행 방법
```powershell
uv sync                                     # 가상환경·패키지 설치 (Python 3.12)
uv run python scripts/build_rone.py         # data/raw 의 R-ONE CSV → data/clean/rone/*.parquet
uv run streamlit run app/Home.py            # http://localhost:8501
uv run pytest                               # 테스트
```
- 원본 CSV(`data/raw/`)와 정리본(`data/clean/`)은 git에 올리지 않습니다.
- 공개 배포에서는 화면이 TiDB Cloud(읽기 전용 계정)에서 데이터를 읽습니다. 로컬에서는 Parquet를 읽습니다.

## 폴더 구조
```
app/                 Streamlit 화면 (Home.py, pages/, common.py = 데이터 읽기·출처 푸터)
src/realestate/      ingest/ (R-ONE 정리), metrics/ (지표), db/ (TiDB 접속·스키마)
scripts/             build_rone.py, publish.py (TiDB 업로드), check_secrets.py (비밀값 검사)
config/sources.yaml  비밀이 아닌 설정 (API 주소·한도)
tests/               정리·지표·권한 검사·화면 규칙 테스트
```

## 보안
- API 키·DB 비밀번호는 `.env`(로컬)와 Streamlit Secrets(공개 앱, 읽기 전용 계정만)에만 둡니다.
- 커밋할 때마다 `scripts/check_secrets.py` 가 비밀값·금지 파일을 검사합니다 (git pre-commit 훅).
- 공개 앱은 접속한 DB 계정에 쓰기 권한이 있으면 실행을 멈춥니다.

## 출처
- 자료: 한국부동산원 R-ONE 부동산통계정보 (reb.or.kr/r-one) — 저작자·출처표시 조건으로 자유 이용
- 자료: 국토교통부 실거래가 공개시스템, 한국자산관리공사 온비드 (MVP 2 이후)

공공 자료를 가공한 것이며 투자 조언이 아닙니다.
