"""매매 점수 평가 보충 집계 → reports/score_eval_detail.md (읽기만: eval 다시 안 함, MLflow 안 씀).

score_backtest.py eval 리포트에 없는 것: 전체 시군구-월 기준 등급 비율 (데이터 부족 포함),
정답이 있는 비율, 절대 불가 규칙별 건수·성과. config/score.json 고정 설정과 data/score 기준월 파일로 다시 셈.

실행: uv run python scripts/score_eval_detail.py
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from score_backtest import CONFIG, PERIODS, ROOT, load

from realestate.score import region

OUT = ROOT / "reports" / "score_eval_detail.md"
ORDER = [*region.GRADES, region.NODATA]
LABEL = {"eval": "평가 (T 2024.1~2025.9)", "check": "확인 (T 2025.10~2026.8, 참고)"}


def pct(v: float, sign: bool = True) -> str:
    return "-" if pd.isna(v) else (f"{v * 100:+.2f}" if sign else f"{v * 100:.1f}")


def counts(sc: pd.DataFrame) -> list[str]:
    lines = ["| 구분 | 시군구-월 | 비율 (%) | 6개월 정답 있음 | 비율 (%) | 12개월 정답 있음 | 비율 (%) |",
             "|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for g in [*ORDER, "합계"]:
        d = sc if g == "합계" else sc[sc["grade"] == g]
        n6, n12 = d["X6"].notna().sum(), d["X12"].notna().sum()
        lines.append(
            f"| {g} | {len(d):,} | {pct(len(d) / len(sc), False)} | {n6:,} | {pct(n6 / len(d) if len(d) else float('nan'), False)} | "
            f"{n12:,} | {pct(n12 / len(d) if len(d) else float('nan'), False)} |"
        )
    return lines


def months_with_label(sc: pd.DataFrame, x: str) -> str:
    t = sorted(sc.loc[sc[x].notna(), "t"].unique())
    return f"{t[0]}~{t[-1]} ({len(t)}개월)" if t else "없음"


def rule_lines(sc: pd.DataFrame) -> list[str]:
    nev = sc["grade"] == region.NEVER
    groups = {
        "절대 불가 전체": nev,
        "R2 거래 급감 + 하락 지속": nev & sc["r2"],
        "R3 해제 급증": nev & sc["r3"],
        "R1 과 겹침 (절대 불가로 둠)": nev & sc["r1"],
        "규칙 없이 절대 불가": nev & ~(sc["r2"] | sc["r3"]),
        "데이터 부족 (R1, 등급 없음)": sc["grade"] == region.NODATA,
    }
    lines = ["| 구분 | 건수 | 6개월 정답 | 6개월 초과 평균 (%) | 적중률 (%) | 12개월 정답 | 12개월 초과 평균 (%) | 12개월 최대 하락 평균 (%) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for k, m in groups.items():
        d = sc[m]
        d6, d12 = d[d["X6"].notna()], d[d["X12"].notna()]
        lines.append(
            f"| {k} | {len(d):,} | {len(d6):,} | {pct(d6['X6'].mean())} | {pct((d6['X6'] > 0).mean(), False) if len(d6) else '-'} | "
            f"{len(d12):,} | {pct(d12['X12'].mean())} | {pct(d12['mdd'].mean())} |"
        )
    by_t = sc[nev].groupby("t").size()
    lines += ["", "절대 불가가 나온 기준월: " + (", ".join(f"{t} {n}건" for t, n in by_t.items()) or "없음")]
    return lines


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    lines = [
        "# 매매 점수 평가 보충 집계",
        "",
        f"- 만든 시각: {datetime.now():%Y-%m-%d %H:%M} · `scripts/score_eval_detail.py` (eval 다시 안 함, 고정 설정으로 등급만 다시 셈)",
        f"- 설정: `config/score.json` {cfg['cand']} {cfg['weights']} (weights_hash `{cfg['weights_hash']}`, rule_version {cfg['rule_version']})",
        "- 본 결과: `reports/score_backtest.md`. 여기 비율은 정답 유무와 관계없이 전체 시군구-월 기준",
        "- 정답 없음 = 같은 단지·면적대 짝 10개 미만, 또는 정답 창 끝이 2026.9 뒤 (12개월은 T ≤ 2025.7, 6개월은 T ≤ 2026.1 까지만)",
        "",
    ]
    for p in ("eval", "check"):
        sc = region.score(load(p), cfg["weights"])
        a, b = PERIODS[p]
        lines += [
            f"## {LABEL[p]}",
            "",
            f"시군구-월 {len(sc):,} (시군구 {sc['sgg_cd'].nunique()} × {sc['t'].nunique()}개월 {a}~{b}). "
            f"6개월 정답 기준월 {months_with_label(sc, 'X6')}, 12개월 {months_with_label(sc, 'X12')}",
            "",
            "### 등급별 건수와 정답 있는 비율",
            "",
            *counts(sc),
            "",
            "### 절대 불가 규칙별·데이터 부족",
            "",
            *rule_lines(sc),
            "",
        ]
    OUT.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
