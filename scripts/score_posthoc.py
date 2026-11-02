"""매매 점수 사후 분석 (평가 결과를 본 뒤 추가. 설정·등급·판정은 바꾸지 않음, eval 다시 안 함, MLflow 안 씀).

- 권역(수도권 서울·인천·경기 / 지방) 안에서 매월 Spearman IC: 점수와 4요소 백분위 각각, 6·12개월
- 95% 구간: 월별 IC 의 원형 블록 부트스트랩 (블록 = h 개월, 1,000회, 본 평가와 같은 방법)
- reports/score_backtest.md 끝에 "사후 분석"·"한계와 다음 버전 과제" 절을 붙임 (다시 실행하면 그 절만 바꿔 씀)

실행: uv run python scripts/score_posthoc.py
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from score_backtest import CAPITAL, CONFIG, REPORT, load

from realestate.score import backtest as bt
from realestate.score import region

MARK = "## 6. 사후 분석"
COLS = {
    "score": "4요소 점수",
    "P_A": "A 고점 대비 하락률",
    "P_B": "B 모델 저평가",
    "P_C": "C 3개월 추세",
    "P_D": "D 거래량 변화",
}
AREAS = {"전국": None, "수도권 안": True, "지방 안": False}
N_BOOT = 1000


def ic_ci(sc: pd.DataFrame, col: str, h: int) -> tuple[float, float, float, float, int]:
    """(IC 평균, 2.5%, 97.5%, 양수 달 비율, 달 수)."""
    ic = bt.monthly_ic(sc, col, f"X{h}")
    v = ic.to_numpy()
    boot = np.array([v[i].mean() for i in bt.block_months(list(ic.index), h, N_BOOT, seed=0)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), float((v > 0).mean()), len(v)


def section(sc: pd.DataFrame) -> list[str]:
    cap = sc["sgg_cd"].str[:2].isin(CAPITAL)
    f3 = lambda v: f"{v:+.3f}"  # noqa: E731
    lines = [
        f"{MARK} (평가 결과를 본 뒤 추가 — 설정·등급·판정은 그대로)",
        "",
        f"- 만든 시각: {datetime.now():%Y-%m-%d %H:%M} · `scripts/score_posthoc.py`. 평가 T 2024.1~2025.9, 고정 설정 그대로",
        "- 권역 안 IC: 매월 같은 권역 시군구끼리만 Spearman(값, 전국 대비 초과 수익) → 월 평균. "
        f"95% 구간은 월별 IC 원형 블록 부트스트랩 (블록 = h 개월, {N_BOOT:,}회). 달마다 시군구 {bt.IC_MIN_N}개 미만이면 뺌",
        "- 요소 IC 는 그 요소의 전국 백분위 (점수에 들어가는 방향 그대로: 클수록 점수가 높음). 권역 안 순위는 원값 순위와 같음",
        "- 사후 분석이므로 판정을 바꾸지 않음. 결과를 보고 고른 비교가 아니도록 요청된 권역 2개와 요소 4개를 모두 적음",
        "",
        f"시군구 수 (기준월 평균): 수도권 {cap.groupby(sc['t']).sum().mean():.0f}, 지방 {(~cap).groupby(sc['t']).sum().mean():.0f}",
        "",
    ]
    for h in (6, 12):
        lines += [
            f"### 권역별 IC — {h}개월 (평균 [95% 구간] / 양수 달 %)",
            "",
            "| 값 | " + " | ".join(AREAS) + " |",
            "|---|" + "---:|" * len(AREAS),
        ]
        for col, name in COLS.items():
            cells = []
            for flag in AREAS.values():
                d = sc if flag is None else sc[cap == flag]
                m, lo, hi, pos, n = ic_ci(d, col, h)
                cells.append(f"{f3(m)} [{f3(lo)} ~ {f3(hi)}] / {pos * 100:.0f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        n = len(bt.monthly_ic(sc, "score", f"X{h}"))
        lines += ["", f"(기준월 {n}개)", ""]

    lines += ["### 권역 구성 (평가 6개월 정답 있는 건)", "",
              "| 권역 | 시군구-월 | 추천 | 고려 | 고려 대상 아님 | 권역 평균 초과 (%) |", "|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    d = sc[sc["X6"].notna()]
    for name, flag in (("수도권", True), ("지방", False)):
        g = d[cap[d.index] == flag]
        vc = g["grade"].value_counts()
        lines.append(
            f"| {name} | {len(g):,} | {vc.get(region.REC, 0):,} | {vc.get(region.CONSIDER, 0):,} | "
            f"{vc.get(region.NOT, 0):,} | {g['X6'].mean() * 100:+.2f} |"
        )
    rec = d[d["grade"] == region.REC]
    lines += [
        "",
        f"추천 중 수도권 비율 {cap[rec.index].mean() * 100:.0f}% (전체 시군구-월 중 수도권 {cap[d.index].mean() * 100:.0f}%)",
        "",
    ]
    return lines


LIMITS = """## 7. 한계와 다음 버전 과제 (사후 정리)

**한계**
- **추천·고려가 구분되지 않음**: 평가 6개월 초과 추천 +0.27% · 고려 +0.26%, 12개월은 고려 +0.69% > 추천 +0.41%. 추천 적중률도 50% 미만 (6개월 48.9%). 판정 ②(추천 − 고려 대상 아님)는 맞았지만 "추천" 등급이 "고려"보다 낫다는 근거는 없음
- **권역 구성 효과**: 추천의 대부분이 수도권이고 평가 기간 수도권이 지방보다 많이 올라, 전체 추천 − 고려 대상 아님 차이(+1.18%p)의 상당 부분이 "수도권을 골랐다"에서 나옴. 권역 안에서는 추천 − 고려 대상 아님이 수도권 −0.51%p · 지방 −0.04%p (3장 표).
  **권역 안 점수 IC 는 0 과 구분 안 됨**: 6개월 수도권 −0.058 [−0.179 ~ +0.046] · 지방 −0.006 [−0.075 ~ +0.067], 12개월 −0.049 · +0.012 (전국 +0.179 · +0.202). 즉 평가 기간 점수의 순위 정보는 거의 다 권역 간 차이였음
- **A·B 단독 IC 음수**: 평가 기간 6개월 "A 만" −0.156, "B 만" −0.341. 권역 안에서도 음수 (A 수도권 −0.594 · 지방 −0.137, B −0.490 · −0.244). 많이 내린 곳·모델보다 싸게 거래된 곳이 이후 덜 오름 (하락 후 반등이 아니라 추세 지속)
  → 양의 IC 는 C·D 에서 옴 (C 수도권 +0.619 · 지방 +0.292, D +0.608 · +0.054). 균등 가중치에서 A·B 가 C·D 를 상쇄해 권역 안 점수 IC 가 0 근처가 됨. 가중치 s1 은 튜닝 기간(2022.2~2023.4 하락기)에서 고른 것이라 국면이 바뀌면 요소 방향이 달라질 수 있음
  → 이 요소별 결과는 평가 기간을 보고 얻은 것이므로 이번 버전 가중치를 바꾸는 근거로 쓰지 않음 (다음 버전에서 따로 튜닝·평가)
- **R2 미검증**: 평가 기간에 하락기가 없어 R2 가 한 번도 안 걸림 (튜닝 기간 452건은 2022.10~2023.3 한 하락기). **R3 표본 부족**: 평가 21건 중 6개월 정답 3건, 12개월 0건
- 데이터 부족 633건(11.8%)은 정답이 있는 비율이 6~8% 라 성과로 판단 못 함. 12개월 정답은 기준월 2025.7 까지만

**다음 버전 과제** (이번 설정·판정은 그대로 두고, 새 버전으로 따로 튜닝·평가)
- **2026.9 이후 데이터로 검증**: 이번 평가·튜닝에 안 쓴 계약월(2026.10~)이 쌓이면 확인 기간(T 2025.10~)의 6·12개월 정답과 새 기준월로 같은 설정을 그대로 다시 채점. 결과를 보고 고치는 다음 버전은 그 뒤의 기간으로 평가
- 추천·고려 구분: 경계 70/50·P_C ≥ 40 을 튜닝 대상으로 할지, 추천을 상위 비율로 정할지
- 권역 구성: 권역(또는 시도) 안 백분위로 점수를 매기는 안, 판정에 "권역 안 IC > 0 (95% 하한 > 0)" 추가
- A·B: 방향·가중치를 시장 국면별로 볼지, 빼고 C·D 만 쓸지 (사전 등록 후 튜닝 기간에서만 결정)
- R2·R3: 하락기가 들어간 기간에서 검증하기 전까지 "검증 안 됨"으로 화면에 표시
"""


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    sc = region.score(load("eval"), cfg["weights"])
    text = REPORT.read_bytes().decode("utf-8").replace("\r\n", "\n")
    base = text.split(MARK)[0].rstrip("\n") + "\n\n"
    out = base + "\n".join(section(sc)) + "\n" + LIMITS
    REPORT.write_bytes(out.encode("utf-8"))
    print(out[len(base) :])


if __name__ == "__main__":
    main()
