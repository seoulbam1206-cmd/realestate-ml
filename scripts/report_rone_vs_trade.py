"""R-ONE 지수 vs 실거래 월별 중위 ㎡당가 비교 (MVP 2 완료 기준).

- 실거래: `realestate.ingest.trades` 규칙으로 읽은 정리본, 해제 거래 제외. 지역 × 월 중위 ㎡당가(만원/㎡)
  → 거래 수가 적은 달의 흔들림을 줄이려고 3개월 이동평균, 2026.6 = 100 으로 지수화 (R-ONE 과 같은 기준월)
- R-ONE: 매매가격지수(조사, survey) · 실거래가격지수(real, 시도·묶음·서울 권역만 있음)
- 지역: 전국·수도권·지방권, 시도 17개, R-ONE 시군구·일반구 (이름으로 시군구 코드에 연결 → region_map)
- 이번 달은 거래가 덜 들어와서 빼고, 지난달은 신고 기간(30일)이 안 끝나 '진행 중' 으로 표시

결과
- data/clean/agg/agg_region_month.parquet : 시군구 × 월 거래 수·해제 수·㎡당가 p25/중위/p75 (PLAN.md 5-3)
- data/clean/reference/region_map.parquet  : R-ONE 지역 → 시군구 코드
- reports/rone_vs_trade.html              : 지역을 골라 보는 그래프 (plotly)
- reports/rone_vs_trade.md                : 지역별 상관(전년 동월 대비 변화율) 요약

실행: uv run python scripts/report_rone_vs_trade.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime

import duckdb
import numpy as np
import pandas as pd

from realestate.config import BJD_RAW, DATA_CLEAN, REFERENCE_CLEAN, RONE_CLEAN, ROOT
from realestate.ingest import bjd, trades
from realestate.ingest.molit_jobs import shift_ym

REPORTS = ROOT / "reports"
AGG_OUT = DATA_CLEAN / "agg" / "agg_region_month.parquet"
MAP_OUT = REFERENCE_CLEAN / "region_map.parquet"
BASE_YM = "202606"
MIN_N_PRICE = 5  # 공개 자료 규칙: 거래가 이보다 적은 칸은 중위가를 비운다 (CLAUDE.md)
SIDO = {  # R-ONE 시도 이름 → (시군구 코드 앞 2자리, 법정동 시도 이름)
    "서울": ("11", "서울특별시"), "부산": ("26", "부산광역시"), "대구": ("27", "대구광역시"),
    "인천": ("28", "인천광역시"), "대전": ("30", "대전광역시"), "울산": ("31", "울산광역시"),
    "세종": ("36", "세종특별자치시"), "경기": ("41", "경기도"), "충북": ("43", "충청북도"),
    "충남": ("44", "충청남도"), "경북": ("47", "경상북도"), "경남": ("48", "경상남도"),
    "제주": ("50", "제주특별자치도"), "강원": ("51", "강원특별자치도"), "전북": ("52", "전북특별자치도"),
    "전남광주": ("12", "전남광주통합특별시"),
}  # fmt: skip
CAPITAL = {"서울", "인천", "경기"}


def region_map(rone_region: pd.DataFrame, sgg_names: dict[str, str]) -> pd.DataFrame:
    """R-ONE 지역 → 시군구 코드 목록 (한 지역에 여러 개 가능). 못 찾은 지역은 sgg_cd 없음으로 남긴다."""
    lawds = list(sgg_names)
    by_prefix = lambda p: [c for c in lawds if c.startswith(p)]  # noqa: E731
    gwangju = [
        c for c in by_prefix("12") if sgg_names[c].endswith("구")
    ]  # 옛 광주광역시 = 통합시 안의 자치구
    rows = []
    for _, r in rone_region.iterrows():
        key, kind = r["region_key"], r["region_type"]
        codes: list[str] = []
        if key == "전국":
            codes = lawds
        elif key == "수도권":
            codes = [c for s in CAPITAL for c in by_prefix(SIDO[s][0])]
        elif key == "지방권":
            codes = [c for s, (p, _) in SIDO.items() if s not in CAPITAL for c in by_prefix(p)]
        elif kind == "시도":
            if key == "전남광주>광주":
                codes = gwangju
            elif key == "전남광주>전남":
                codes = [c for c in by_prefix("12") if c not in gwangju]
            elif key in SIDO:
                codes = by_prefix(SIDO[key][0])
        elif kind in ("시군구", "일반구"):
            sido = key.split(">")[0]
            rest = r["display_name"].split(" ", 1)[1]
            full = f"{SIDO[sido][1]} {rest}" if sido in SIDO else None
            codes = [c for c, n in sgg_names.items() if full and (n == full or n.startswith(full + " "))]
        if codes:
            rows += [{"region_key": key, "region_type": kind, "sgg_cd": c} for c in codes]
        elif kind in ("시도", "시군구", "일반구"):  # 권역·그 밖의 묶음은 비교하지 않음
            rows.append({"region_key": key, "region_type": kind, "sgg_cd": None})
    return pd.DataFrame(rows)


def build_agg(con: duckdb.DuckDBPyConnection, files: list) -> pd.DataFrame:
    paths = ", ".join(f"'{p.as_posix()}'" for p in files)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE t AS
        SELECT sgg_cd, strftime(deal_date, '%Y%m') AS ym, price_per_m2, is_canceled
        FROM read_parquet([{paths}], union_by_name = true)
        """
    )
    return con.execute(
        """
        SELECT sgg_cd, ym, count(*) FILTER (WHERE NOT is_canceled) AS trade_cnt,
               count(*) FILTER (WHERE is_canceled) AS cancel_cnt,
               quantile_cont(price_per_m2, 0.25) FILTER (WHERE NOT is_canceled) AS p25_ppm2,
               median(price_per_m2) FILTER (WHERE NOT is_canceled) AS median_ppm2,
               quantile_cont(price_per_m2, 0.75) FILTER (WHERE NOT is_canceled) AS p75_ppm2
        FROM t GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).df()


def region_medians(con: duckdb.DuckDBPyConnection, rmap: pd.DataFrame) -> pd.DataFrame:
    con.register("rmap", rmap.dropna(subset=["sgg_cd"]))
    return con.execute(
        """
        SELECT m.region_key, t.ym, count(*) AS n, median(t.price_per_m2) AS med
        FROM t JOIN rmap m USING (sgg_cd) WHERE NOT t.is_canceled GROUP BY 1, 2
        """
    ).df()


def _rolling_index(df: pd.DataFrame, key: str, value: str) -> pd.Series:
    """key 별 3개월 이동평균을 기준월 = 1 로 나눈 값 (df 는 key × 전체 월 격자, 월 순서)."""
    m3 = df.groupby(key)[value].transform(lambda x: x.rolling(3, min_periods=2).mean())
    base = m3[df["ym"] == BASE_YM].groupby(df.loc[df["ym"] == BASE_YM, key]).first()
    return m3 / df[key].map(base)


def fixed_weight_index(agg: pd.DataFrame, rmap: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """지역 × 월 고정 가중 지수: 시군구별 (중위 ㎡당가 ÷ 기준월) 을 기간 전체 거래 수로 가중평균 × 100.

    지역 단순 중위가는 그 달 어느 시군구에서 많이 팔렸는지(서울 비중 등)에 따라 움직이므로,
    시군구 구성을 고정해 가격 변화만 남긴다 (2026-10-30: 전국 단순 중위가는 조사 지수와 상관 -0.76).
    """
    idx = pd.MultiIndex.from_product([agg["sgg_cd"].unique(), months], names=["sgg_cd", "ym"])
    g = agg.set_index(["sgg_cd", "ym"]).reindex(idx).reset_index()
    g["ratio"] = _rolling_index(g, "sgg_cd", "median_ppm2")
    weight = agg.groupby("sgg_cd")["trade_cnt"].sum()
    g = g.dropna(subset=["ratio"]).assign(w=lambda x: x["sgg_cd"].map(weight))
    m = g.merge(rmap.dropna(subset=["sgg_cd"])[["region_key", "sgg_cd"]], on="sgg_cd")
    m["wr"] = m["w"] * m["ratio"]
    out = m.groupby(["region_key", "ym"])[["wr", "w"]].sum()
    return (out["wr"] / out["w"] * 100).rename("trade_idx").reset_index()


def series(med: pd.DataFrame, fixed: pd.DataFrame, rone: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """지역 × 월: trade_idx(고정 가중), raw_idx(지역 단순 중위), survey, real (모두 2026.6 = 100)."""
    idx = pd.MultiIndex.from_product([med["region_key"].unique(), months], names=["region_key", "ym"])
    s = med.set_index(["region_key", "ym"]).reindex(idx).reset_index()
    s["raw_idx"] = _rolling_index(s, "region_key", "med") * 100
    s = s.merge(fixed, on=["region_key", "ym"], how="left")
    r = rone.assign(ym=rone["ym"].dt.strftime("%Y%m")).pivot_table(
        index=["region_key", "ym"], columns="source", values="index_value"
    )
    return s.merge(r.reset_index(), on=["region_key", "ym"], how="left")


def yoy_corr(g: pd.DataFrame, col: str, own: str = "trade_idx") -> float:
    a = g[own].pct_change(12, fill_method=None)
    b = g[col].pct_change(12, fill_method=None) if col in g else pd.Series(dtype=float)
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 12 else float("nan")


def summarize(s: pd.DataFrame, regions: pd.DataFrame, upto: str) -> pd.DataFrame:
    rows = []
    for key, g in s[s["ym"] <= upto].groupby("region_key", sort=False):
        g = g.sort_values("ym")
        first = g.dropna(subset=["trade_idx", "survey"]).head(1)
        last = g.dropna(subset=["trade_idx", "survey"]).tail(1)
        rows.append(
            {
                "region_key": key,
                "n_avg": g["n"].mean(),
                "corr_survey": yoy_corr(g, "survey"),
                "corr_real": yoy_corr(g, "real"),
                "corr_raw": yoy_corr(g, "survey", own="raw_idx"),
                "chg_trade": (last["trade_idx"].iloc[0] / first["trade_idx"].iloc[0] - 1) * 100
                if len(first)
                else np.nan,
                "chg_survey": (last["survey"].iloc[0] / first["survey"].iloc[0] - 1) * 100
                if len(first)
                else np.nan,
                "from": first["ym"].iloc[0] if len(first) else None,
                "to": last["ym"].iloc[0] if len(last) else None,
            }
        )
    return pd.DataFrame(rows).merge(regions, on="region_key", how="left")


def write_md(summ: pd.DataFrame, s: pd.DataFrame, rmap: pd.DataFrame, upto: str, partial: str) -> None:
    top = summ[summ["region_type"].isin(["묶음", "시도"])].copy()
    top["o"] = top["region_key"].map({"전국": 0, "수도권": 1, "지방권": 2}).fillna(3)
    top = top.sort_values(["o", "n_avg"], ascending=[True, False])
    sgg = summ[summ["region_type"].isin(["시군구", "일반구"])]
    unmapped = rmap[rmap["sgg_cd"].isna()]["region_key"].tolist()
    f = lambda v, p="{:.2f}": "-" if pd.isna(v) else p.format(v)  # noqa: E731
    sido = s[s["region_key"].isin(top.loc[top["region_type"] == "시도", "region_key"]) & (s["ym"] <= upto)]
    min_sido_n = int(sido["n"].min())
    max_jump = float(sido.groupby("region_key")["trade_idx"].pct_change(fill_method=None).abs().max() * 100)
    lines = [
        "# R-ONE 지수 vs 실거래 중위 ㎡당가",
        "",
        f"- 만든 시각: {datetime.now():%Y-%m-%d %H:%M} · 스크립트 `scripts/report_rone_vs_trade.py` · 그래프 `reports/rone_vs_trade.html`",
        f"- 실거래 지수: 시군구별 해제 제외 중위 ㎡당가(3개월 이동평균) ÷ {BASE_YM[:4]}.{BASE_YM[4:]} 값을 기간 전체 거래 수로 가중평균 × 100 (고정 가중)",
        "- 단순 중위: 지역 전체 거래를 한데 모은 중위 ㎡당가 지수. 어느 시군구에서 많이 팔렸는지(구성)에 따라 움직임 → 비교용",
        f"- 비교 끝 월 {upto} ({partial} 은 신고 진행 중이라 그래프에만 표시)",
        "- 상관: 전년 동월 대비 변화율끼리의 상관계수 (12개월 이상 겹칠 때). 1에 가까울수록 같은 방향·같은 시점에 움직임",
        "",
        "## 전국·묶음·시도",
        "",
        "| 지역 | 월평균 거래 | 상관(조사 지수) | 상관(실거래가격지수) | 단순 중위 상관(조사) | 실거래 변화 % | 조사 지수 변화 % | 기간 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['display_name']} | {r['n_avg']:,.0f} | {f(r['corr_survey'])} | {f(r['corr_real'])} | "
            f"{f(r['corr_raw'])} | {f(r['chg_trade'], '{:+.1f}')} | {f(r['chg_survey'], '{:+.1f}')} | {r['from']}~{r['to']} |"
        )
    q = sgg["corr_survey"].describe(percentiles=[0.25, 0.5, 0.75])
    small = sgg[sgg["n_avg"] < 30]
    lines += [
        "",
        f"## 시군구·일반구 ({len(sgg)}개)",
        "",
        f"- 상관(조사 지수) 분포: 하위 25% {q['25%']:.2f} · 중앙값 {q['50%']:.2f} · 상위 25% {q['75%']:.2f}",
        f"- 월평균 거래 30건 미만 지역 {len(small)}개의 상관 중앙값 {small['corr_survey'].median():.2f} "
        f"(30건 이상 {sgg[sgg['n_avg'] >= 30]['corr_survey'].median():.2f}) → 거래가 적을수록 중위가가 단지 구성에 따라 흔들림",
        "",
        "상관이 낮은 지역 (하위 10)",
        "",
        "| 지역 | 월평균 거래 | 상관(조사 지수) | 실거래 변화 % | 조사 지수 변화 % |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {r['display_name']} | {r['n_avg']:,.0f} | {f(r['corr_survey'])} | {f(r['chg_trade'], '{:+.1f}')} | {f(r['chg_survey'], '{:+.1f}')} |"
            for _, r in sgg.sort_values("corr_survey").head(10).iterrows()
        ],
        "",
        "## 해석",
        "",
        f"- 전국 단순 중위가의 상관은 {f(top.set_index('region_key').loc['전국', 'corr_raw'])}, 고정 가중은 {f(top.set_index('region_key').loc['전국', 'corr_survey'])}: "
        "2022년 거래 절벽 때 서울·수도권 비중이 줄어 단순 중위가가 내려가고 2023년 다시 오르는 등, 지역 구성이 가격 변화를 가림",
        "- 시군구 안에서도 중위 ㎡당가는 '그 달에 팔린 집'의 구성(신축·대단지 비중)에 따라 움직여, 같은 집의 가격 변화를 재는 지수보다 흔들림이 큼",
        "  → MVP 3 시세 모델은 단지·면적·층·연식을 통제해 이 구성 효과를 걷어내는 것이 목적",
        "- 수도권·경기·인천·서울(실거래가격지수)처럼 가격이 크게 오르내린 곳은 상관 0.85~0.97 → 실거래 정리본의 흐름이 공식 지수와 맞음",
        "- 지방은 조사 지수가 6년간 거의 제자리(지방권 +2.3%)라 비교할 신호가 작고, 신축 비중이 늘며 ㎡당 중위가만 오른 효과가 커서 상관이 낮거나 음수.",
        f"  누락 문제라면 특정 달 거래가 비거나 지수가 크게 꺾여야 함 → 시도 중 월 거래가 가장 적은 달도 {min_sido_n:,}건, "
        f"고정 가중 지수의 한 달 변동은 가장 큰 것이 {max_jump:.1f}% (그래프에서 지역별로 확인 가능)",
        f"- R-ONE 지역 중 시군구 코드에 연결 못 한 것: {len(unmapped)}개"
        + (f" ({', '.join(unmapped[:10])})" if unmapped else ""),
        "",
    ]
    (REPORTS / "rone_vs_trade.md").write_text("\n".join(lines), encoding="utf-8")


def _num(v) -> float | None:
    return None if pd.isna(v) else round(float(v), 2)


def write_html(s: pd.DataFrame, regions: pd.DataFrame, summ: pd.DataFrame, partial: str) -> None:
    order = {"묶음": 0, "시도": 1, "시군구": 2, "일반구": 3}
    regs = regions[regions["region_key"].isin(s["region_key"])].copy()
    regs["o"] = regs["region_type"].map(order)
    regs = regs.sort_values(["o"], kind="stable")
    corr = summ.set_index("region_key")[["corr_survey", "corr_real", "corr_raw", "n_avg"]]
    data = {}
    for key, g in s.groupby("region_key"):
        g = g.sort_values("ym")
        data[key] = {
            "ym": [f"{y[:4]}-{y[4:]}" for y in g["ym"]],
            "trade": [_num(v) for v in g["trade_idx"]],
            "raw": [_num(v) for v in g["raw_idx"]],
            "survey": [_num(v) for v in g["survey"]] if "survey" in g else [],
            "real": [_num(v) for v in g["real"]] if "real" in g else [],
            "n": [None if pd.isna(v) else int(v) for v in g["n"]],
            "med": [
                None if pd.isna(n) or n < MIN_N_PRICE else _num(v) for n, v in zip(g["n"], g["med"])
            ],
            "corr_survey": _num(corr.loc[key, "corr_survey"]),
            "corr_real": _num(corr.loc[key, "corr_real"]),
            "corr_raw": _num(corr.loc[key, "corr_raw"]),
        }
    options = [
        {"key": k, "name": n, "group": t}
        for k, n, t in regs[["region_key", "display_name", "region_type"]].itertuples(index=False)
    ]
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace(
        "__OPTIONS__", json.dumps(options, ensure_ascii=False)
    )
    html = html.replace("__PARTIAL__", f"{partial[:4]}-{partial[4:]}").replace(
        "__BASE__", f"{BASE_YM[:4]}.{BASE_YM[4:]}"
    )
    html = html.replace("__MADE__", f"{datetime.now():%Y-%m-%d %H:%M}")
    (REPORTS / "rone_vs_trade.html").write_text(html, encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    codes = bjd.load(BJD_RAW)
    lawds = bjd.lawd_codes(codes)
    sgg = codes[codes["is_active"] & codes["bjd_cd"].str.endswith("00000")]
    names = dict(zip(sgg["sgg_cd"], sgg["full_nm"], strict=True))
    sgg_names = {c: names[c] for c in lawds}
    src = trades.month_sources(lawds)
    this_ym = f"{date.today():%Y%m}"
    src = src[src["ym"] < this_ym]  # 이번 달은 거래가 덜 들어옴
    files = [p for f in src["files"] for p in f]
    partial, upto = shift_ym(this_ym, -1), shift_ym(this_ym, -2)

    rone_region = pd.read_parquet(RONE_CLEAN / "rone_region.parquet")
    rone = pd.read_parquet(RONE_CLEAN / "rone_index.parquet")
    rmap = region_map(rone_region, sgg_names)
    MAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    rmap.to_parquet(MAP_OUT, index=False)

    con = duckdb.connect()
    agg = build_agg(con, files)
    AGG_OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(AGG_OUT, index=False)
    med = region_medians(con, rmap)
    fixed = fixed_weight_index(agg, rmap, sorted(src["ym"]))
    s = series(med, fixed, rone, sorted(src["ym"]))
    regions = rone_region[["region_key", "region_type", "display_name"]]
    summ = summarize(s, regions, upto)
    REPORTS.mkdir(exist_ok=True)
    write_md(summ, s, rmap, upto, partial)
    write_html(s, regions, summ, partial)
    top = summ[summ["region_key"].isin(["전국", "수도권", "지방권", "서울"])]
    print(
        top[["region_key", "corr_survey", "corr_real", "corr_raw", "chg_trade", "chg_survey"]]
        .round(2)
        .to_string(index=False)
    )
    print(
        f"시군구 × 월 {len(agg):,}행 → {AGG_OUT.relative_to(ROOT)}; 지역 {rmap['region_key'].nunique()}개 "
        f"(연결 못 함 {int(rmap['sgg_cd'].isna().sum())}) → reports/rone_vs_trade.html, .md"
    )


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>R-ONE vs 실거래</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --text: #0b0b0b; --text-2: #52514e; --muted: #8a8984; --grid: #e7e6e2; --border: #d9d8d3;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --text: #f0efec; --text-2: #c3c2b7; --muted: #8f8e86; --grid: #2e2e2c; --border: #3a3a37;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --text: #f0efec; --text-2: #c3c2b7; --muted: #8f8e86; --grid: #2e2e2c; --border: #3a3a37;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--text);
  font: 15px/1.55 -apple-system, "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--text-2); margin: 0 0 20px; font-size: 14px; }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 8px; }
select { font: inherit; padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px;
  background: var(--surface); color: var(--text); max-width: 100%; }
.stats { color: var(--text-2); font-size: 14px; }
.stats b { color: var(--text); font-variant-numeric: tabular-nums; }
#chart { width: 100%; min-width: 0; height: 440px; overflow: hidden; }
.controls > * { min-width: 0; }
details { margin-top: 16px; }
summary { cursor: pointer; color: var(--text-2); }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; margin-top: 8px; }
th, td { padding: 4px 10px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
.note { color: var(--muted); font-size: 13px; margin-top: 16px; }
</style>
</head>
<body>
<main>
  <h1>R-ONE 지수 vs 실거래 중위 ㎡당가</h1>
  <p class="sub">모든 선은 __BASE__ = 100 으로 맞춘 지수입니다. 실거래 지수는 시군구별 중위 ㎡당가(해제 제외, 3개월 이동평균)를 거래 수로 고정 가중한 값이고,
  점선은 지역 전체를 한데 모은 단순 중위가입니다 (어느 시군구에서 많이 팔렸는지에 따라 움직임).</p>
  <div class="controls">
    <select id="region" aria-label="지역"></select>
    <span class="stats" id="stats"></span>
  </div>
  <div id="chart" role="img" aria-label="지역별 지수 비교 그래프"></div>
  <details>
    <summary>표로 보기</summary>
    <div class="tablewrap"><table id="table"></table></div>
  </details>
  <p class="note">__PARTIAL__ 은 신고 기간(계약 후 30일)이 끝나지 않아 거래가 덜 들어온 달입니다. 실거래가격지수는 시도·묶음 지역에만 있습니다.
  출처: 한국부동산원 R-ONE, 국토교통부 실거래가 공개시스템·공공데이터포털. 투자 조언이 아닙니다. 만든 시각 __MADE__</p>
</main>
<script>
const DATA = __DATA__;
const OPTIONS = __OPTIONS__;
const PARTIAL = "__PARTIAL__";
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const sel = document.getElementById("region");
const groups = {};
for (const o of OPTIONS) (groups[o.group] ||= []).push(o);
for (const [g, items] of Object.entries(groups)) {
  const og = document.createElement("optgroup"); og.label = g;
  for (const o of items) { const op = document.createElement("option"); op.value = o.key; op.textContent = o.name; og.appendChild(op); }
  sel.appendChild(og);
}
let saved = null; try { saved = localStorage.getItem("region"); } catch (e) {}
sel.value = (saved && DATA[saved]) ? saved : "전국";

function draw() {
  const d = DATA[sel.value];
  try { localStorage.setItem("region", sel.value); } catch (e) {}
  const series = [
    { name: "실거래 지수 (시군구 고정 가중)", y: d.trade, color: css("--s1"), dash: "solid" },
    { name: "R-ONE 매매가격지수 (조사)", y: d.survey, color: css("--s2"), dash: "solid" },
    { name: "R-ONE 실거래가격지수", y: d.real, color: css("--s3"), dash: "solid" },
    { name: "단순 중위 (구성 효과 포함)", y: d.raw, color: css("--muted"), dash: "dot" },
  ].filter((s) => s.y.some((v) => v !== null));
  // 거래 수는 축에 그리지 않고 첫 계열의 호버에만 붙임 (축이 0~10만으로 늘어나지 않게)
  const traces = series.map((s, i) => ({
    x: d.ym, y: s.y, name: s.name, type: "scatter", mode: "lines",
    line: { color: s.color, width: 2, dash: s.dash }, connectgaps: false,
    customdata: d.n,
    hovertemplate: "%{y:.1f}" + (i === 0 ? " · 거래 %{customdata:,}건" : "") + "<extra>" + s.name + "</extra>",
  }));
  const layout = {
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: { color: css("--text-2"), family: getComputedStyle(document.body).fontFamily, size: 12 },
    autosize: true, margin: { l: 44, r: 12, t: 8, b: 40 }, hovermode: "x unified",
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--border"), font: { color: css("--text") } },
    legend: { orientation: "h", y: -0.14, font: { color: css("--text") } },
    xaxis: { gridcolor: css("--grid"), linecolor: css("--border"), showspikes: true, spikemode: "across",
      spikecolor: css("--muted"), spikethickness: 1, spikedash: "dot", hoverformat: "%Y.%m" },
    yaxis: { gridcolor: css("--grid"), zeroline: false, title: { text: "지수", font: { size: 12 } } },
    shapes: [{ type: "rect", xref: "x", yref: "paper", x0: PARTIAL, x1: d.ym[d.ym.length - 1], y0: 0, y1: 1,
      fillcolor: css("--grid"), opacity: 0.5, line: { width: 0 }, layer: "below" }],
  };
  Plotly.react("chart", traces, layout, { displayModeBar: false, responsive: true });
  const nAvg = d.n.filter((v) => v !== null);
  const avg = nAvg.length ? Math.round(nAvg.reduce((a, b) => a + b, 0) / nAvg.length) : 0;
  const fmt = (v) => (v === null ? "-" : v.toFixed(2));
  document.getElementById("stats").innerHTML =
    `월평균 거래 <b>${avg.toLocaleString()}</b>건 · 전년 대비 변화율 상관 (실거래 지수 기준): 조사 지수 <b>${fmt(d.corr_survey)}</b>` +
    (d.real.some((v) => v !== null) ? ` · 실거래가격지수 <b>${fmt(d.corr_real)}</b>` : "") +
    ` · 단순 중위는 <b>${fmt(d.corr_raw)}</b>`;
  const rows = d.ym.map((m, i) => `<tr><td>${m}</td><td>${d.n[i] ?? "-"}</td><td>${d.med[i] === null ? "-" : d.med[i].toFixed(0)}</td>` +
    `<td>${d.trade[i] === null ? "-" : d.trade[i].toFixed(1)}</td><td>${d.raw[i] === null ? "-" : d.raw[i].toFixed(1)}</td>` +
    `<td>${d.survey[i] === null ? "-" : d.survey[i].toFixed(1)}</td>` +
    `<td>${d.real[i] === null ? "-" : d.real[i].toFixed(1)}</td></tr>`).reverse().join("");
  document.getElementById("table").innerHTML =
    "<thead><tr><th>월</th><th>거래</th><th>중위 ㎡당가(만원)</th><th>실거래 지수</th><th>단순 중위 지수</th><th>조사 지수</th><th>실거래가격지수</th></tr></thead><tbody>" + rows + "</tbody>";
}
sel.addEventListener("change", draw);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
draw();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
