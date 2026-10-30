"""실거래 정리본 품질 검사: 중복, 건수 급변, 가격·값 이상치.

읽는 범위: `realestate.ingest.trades` 규칙대로 달마다 CSV 또는 API 한쪽 (섞지 않음). DuckDB 로 Parquet 직접 조회.

검사
1. 중복   — trade_key 중복(있으면 오류), 내용이 똑같은 쌍둥이 거래(정보: 순번으로 구분돼 있음)
2. 건수   — 시군구 × 월 거래 수가 직전 12개월 중앙값의 3배 초과 또는 1/3 미만 (중앙값 20건 이상만).
            최근 2개월은 신고 기간이라 '적음'은 따로 표시. 전국 월별 건수와 전월 대비 변화
3. 이상치 — 값 범위: 금액 ≤ 0, 전용면적 < 5㎡ 또는 > 400㎡, 층 < -5 또는 > 120, 건축년도 < 1950 또는 계약년도+3 초과
            (하한 5㎡: 9㎡대 초소형 호실이 같은 단지에서 해마다 정상 가격으로 거래됨, 2026-10-30 확인)
            가격: 같은 단지·같은 면적(정수 ㎡)·같은 해 거래(해제 제외, 5건 이상) 중앙값 대비 ㎡당가 0.5배 미만 또는 2배 초과
                  시군구 × 연도 안에서 log(㎡당가) 강건 z(중앙값·MAD) 절대값 5 초과
4. 매칭   — 법정동 코드를 못 붙인 거래, 건축년도 없음

결과: reports/quality.md (건수 집계만), data/clean/quality/flags.parquet (걸린 거래 목록, 로컬 전용)
종료 코드: 1 = trade_key 중복 또는 값 범위 오류가 있음

실행: uv run python scripts/check_quality.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime

import duckdb
import pandas as pd

from realestate.config import BJD_RAW, DATA_CLEAN, ROOT
from realestate.ingest import bjd, trades

REPORTS = ROOT / "reports"
FLAGS_OUT = DATA_CLEAN / "quality" / "flags.parquet"
SPIKE_RATIO, SPIKE_MIN_MEDIAN = 3.0, 20
PRICE_RATIO_LOW, PRICE_RATIO_HIGH, PRICE_MIN_N = 0.5, 2.0, 5
ROBUST_Z = 5.0


def load(con: duckdb.DuckDBPyConnection, files: list) -> None:
    paths = ", ".join(f"'{p.as_posix()}'" for p in files)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW t AS
        SELECT trade_key, source, sgg_cd, bjd_cd, complex_key, apt_nm, exclu_use_ar, floor, build_year,
               deal_date, strftime(deal_date, '%Y%m') AS ym, year(deal_date) AS yr, deal_amount, price_per_m2,
               is_canceled, apt_dong, bonbun, bubun
        FROM read_parquet([{paths}], union_by_name = true)
        """
    )


def check_duplicates(con) -> dict:
    dup_keys = con.execute(
        "SELECT count(*) FROM (SELECT trade_key FROM t GROUP BY 1 HAVING count(*) > 1)"
    ).fetchone()[0]
    twins = con.execute(
        "SELECT count(*), coalesce(sum(n), 0) FROM (SELECT split_part(trade_key, '-', 1) AS b, count(*) AS n "
        "FROM t GROUP BY 1 HAVING count(*) > 1)"
    ).fetchone()
    return {"dup_trade_key": dup_keys, "twin_groups": twins[0], "twin_rows": int(twins[1])}


def check_counts(con, recent: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = con.execute(
        """
        SELECT ym, count(*) FILTER (WHERE NOT is_canceled) AS trades, count(*) FILTER (WHERE is_canceled) AS canceled
        FROM t GROUP BY 1 ORDER BY 1
        """
    ).df()
    monthly["mom_pct"] = monthly["trades"].pct_change() * 100
    spikes = con.execute(
        f"""
        WITH m AS (
            SELECT sgg_cd, ym, count(*) FILTER (WHERE NOT is_canceled) AS n FROM t GROUP BY 1, 2
        ), w AS (
            SELECT *, median(n) OVER (PARTITION BY sgg_cd ORDER BY ym ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS med,
                      count(*) OVER (PARTITION BY sgg_cd ORDER BY ym ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS k
            FROM m
        )
        SELECT sgg_cd, ym, n, med, n / med AS ratio FROM w
        WHERE k >= 6 AND med >= {SPIKE_MIN_MEDIAN} AND (n > med * {SPIKE_RATIO} OR n < med / {SPIKE_RATIO})
        ORDER BY ym, sgg_cd
        """
    ).df()
    spikes["kind"] = spikes["ratio"].map(lambda r: "많음" if r > 1 else "적음")
    spikes.loc[(spikes["kind"] == "적음") & spikes["ym"].isin(recent), "kind"] = "적음(신고 기간)"
    return monthly, spikes


def check_values(con) -> pd.DataFrame:
    return con.execute(
        """
        SELECT trade_key, sgg_cd, ym, 'value_range' AS flag,
               concat_ws(', ',
                   CASE WHEN deal_amount <= 0 THEN 'amount<=0' END,
                   CASE WHEN exclu_use_ar < 5 OR exclu_use_ar > 400 THEN 'area' END,
                   CASE WHEN floor < -5 OR floor > 120 THEN 'floor' END,
                   CASE WHEN build_year < 1950 OR build_year > yr + 3 THEN 'build_year' END) AS detail,
               price_per_m2, NULL::DOUBLE AS ref
        FROM t
        WHERE deal_amount <= 0 OR exclu_use_ar < 5 OR exclu_use_ar > 400 OR floor < -5 OR floor > 120
           OR build_year < 1950 OR build_year > yr + 3
        """
    ).df()


def check_prices(con) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH ok AS (SELECT * FROM t WHERE NOT is_canceled AND deal_amount > 0 AND exclu_use_ar >= 5),
        cx AS (
            SELECT complex_key, round(exclu_use_ar) AS ar, yr, median(price_per_m2) AS med, count(*) AS n
            FROM ok GROUP BY 1, 2, 3 HAVING count(*) >= {PRICE_MIN_N}
        ),
        sy AS (
            SELECT sgg_cd, yr, median(ln(price_per_m2)) AS med FROM ok GROUP BY 1, 2
        ),
        sy2 AS (
            SELECT ok.sgg_cd, ok.yr, any_value(sy.med) AS med, median(abs(ln(ok.price_per_m2) - sy.med)) AS mad
            FROM ok JOIN sy USING (sgg_cd, yr) GROUP BY 1, 2
        )
        SELECT ok.trade_key, ok.sgg_cd, ok.ym, 'complex_ratio' AS flag,
               printf('%.2f배 (단지·면적·연도 {PRICE_MIN_N}건+ 중앙값 대비)', ok.price_per_m2 / cx.med) AS detail,
               ok.price_per_m2, cx.med AS ref
        FROM ok JOIN cx ON ok.complex_key = cx.complex_key AND round(ok.exclu_use_ar) = cx.ar AND ok.yr = cx.yr
        WHERE ok.price_per_m2 < cx.med * {PRICE_RATIO_LOW} OR ok.price_per_m2 > cx.med * {PRICE_RATIO_HIGH}
        UNION ALL
        SELECT ok.trade_key, ok.sgg_cd, ok.ym, 'robust_z' AS flag,
               printf('z=%.1f (시군구·연도)', (ln(ok.price_per_m2) - sy2.med) / (1.4826 * sy2.mad)) AS detail,
               ok.price_per_m2, exp(sy2.med) AS ref
        FROM ok JOIN sy2 USING (sgg_cd, yr)
        WHERE sy2.mad > 0 AND abs(ln(ok.price_per_m2) - sy2.med) / (1.4826 * sy2.mad) > {ROBUST_Z}
        """
    ).df()


def check_matching(con) -> dict:
    r = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE bjd_cd IS NULL OR sgg_cd IS NULL), "
        "count(*) FILTER (WHERE build_year IS NULL), count(*) FILTER (WHERE is_canceled) FROM t"
    ).fetchone()
    return {"rows": r[0], "unmatched_bjd": r[1], "no_build_year": r[2], "canceled": r[3]}


def _table(df: pd.DataFrame, cols: dict[str, str], fmt: dict | None = None) -> list[str]:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols.values()) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    codes = bjd.load(BJD_RAW)
    lawds = bjd.lawd_codes(codes)
    sgg = codes[codes["is_active"] & codes["bjd_cd"].str.endswith("00000")]
    sgg_nm = dict(zip(sgg["sgg_cd"], sgg["full_nm"], strict=True))
    src = trades.month_sources(lawds)
    files = [p for f in src["files"] for p in f]
    this_ym = f"{date.today():%Y%m}"
    recent = {this_ym, f"{(pd.Period(this_ym, 'M') - 1).strftime('%Y%m')}"}

    con = duckdb.connect()
    load(con, files)
    dup = check_duplicates(con)
    match = check_matching(con)
    monthly, spikes = check_counts(con, recent)
    values = check_values(con)
    prices = check_prices(con)
    flags = pd.concat([values, prices], ignore_index=True)
    FLAGS_OUT.parent.mkdir(parents=True, exist_ok=True)
    flags.assign(checked_at=datetime.now()).to_parquet(FLAGS_OUT, index=False)

    by_flag = flags.groupby("flag").agg(rows=("trade_key", "size"), trades=("trade_key", "nunique"))
    n_flagged = flags["trade_key"].nunique()
    src_counts = src["use"].value_counts().to_dict()
    lines = [
        "# 실거래 정리본 품질 검사",
        "",
        f"- 만든 시각: {datetime.now():%Y-%m-%d %H:%M} · 스크립트 `scripts/check_quality.py`",
        f"- 범위: {src['ym'].min()} ~ {src['ym'].max()} ({len(src)}개월, 읽은 출처 "
        + ", ".join(f"{k} {v}개월" for k, v in sorted(src_counts.items()))
        + f"), 거래 {match['rows']:,}건 (해제 {match['canceled']:,})",
        "",
        "## 1. 중복",
        "",
        f"- trade_key 중복: **{dup['dup_trade_key']:,}** {'✅' if dup['dup_trade_key'] == 0 else '❌'}",
        f"- 내용이 똑같은 쌍둥이 거래: {dup['twin_groups']:,}묶음 {dup['twin_rows']:,}건 (순번 -0, -1… 로 구분, 정상)",
        "",
        "## 2. 건수 급변",
        "",
        f"시군구 × 월 해제 제외 거래 수가 직전 12개월 중앙값의 {SPIKE_RATIO:g}배 초과 또는 1/{SPIKE_RATIO:g} 미만"
        f" (중앙값 {SPIKE_MIN_MEDIAN}건 이상, 직전 6개월 이상 있는 경우만): **{len(spikes)}건**",
        "",
    ]
    if len(spikes):
        k = spikes["kind"].value_counts()
        lines += [", ".join(f"{a} {b}건" for a, b in k.items()), ""]
        show = spikes.assign(nm=spikes["sgg_cd"].map(sgg_nm).fillna("")).sort_values("ratio", ascending=False)
        lines += _table(
            pd.concat([show.head(10), show.tail(5)]).drop_duplicates(),
            {
                "ym": "월",
                "nm": "시군구",
                "n": "거래",
                "med": "직전 12개월 중앙값",
                "ratio": "배율",
                "kind": "구분",
            },
            {"med": "{:.0f}", "ratio": "{:.2f}"},
        )
    lines += ["", "전국 월별 거래 수 (해제 제외, 최근 14개월)", ""]
    lines += _table(
        monthly.tail(14),
        {"ym": "월", "trades": "거래", "canceled": "해제", "mom_pct": "전월 대비 %"},
        {"trades": "{:,}", "canceled": "{:,}", "mom_pct": "{:+.1f}"},
    )
    lines += [
        "",
        "## 3. 이상치",
        "",
        f"걸린 거래 {n_flagged:,}건 ({n_flagged / match['rows']:.3%}). 목록은 `data/clean/quality/flags.parquet` (로컬 전용)",
        "",
        "| 검사 | 걸린 행 | 거래 수 |",
        "|---|---:|---:|",
        *[f"| `{f}` | {r['rows']:,} | {r['trades']:,} |" for f, r in by_flag.iterrows()],
        "",
    ]
    if len(values):
        vd = values["detail"].str.split(", ").explode().value_counts()
        lines += ["값 범위 위반 내역: " + ", ".join(f"{a} {b:,}건" for a, b in vd.items()), ""]
    if len(prices):
        top = (
            prices.groupby("sgg_cd")["trade_key"]
            .nunique()
            .sort_values(ascending=False)
            .head(10)
            .rename("n")
            .reset_index()
        )
        top["nm"] = top["sgg_cd"].map(sgg_nm).fillna("")
        lines += ["가격 이상치가 많은 시군구 (상위 10)", ""]
        lines += _table(top, {"nm": "시군구", "sgg_cd": "코드", "n": "거래 수"}, {"n": "{:,}"})
    lines += [
        "",
        "## 4. 매칭",
        "",
        f"- 법정동 코드 없음: {match['unmatched_bjd']:,}건",
        f"- 건축년도 없음: {match['no_build_year']:,}건",
        "",
        "## 처리 방침",
        "",
        "- trade_key 중복·값 범위 위반이 있으면 종료 코드 1 → 원인을 고치기 전에는 모델 학습에 쓰지 않음",
        "- 가격 이상치는 지우지 않고 표시만 (특수관계 거래·지분 거래일 수 있음). MVP 3 학습에서 `flags.parquet` 로 제외하고 제외 전후 오차를 비교",
        "- 건수 급변 '많음'은 대단지 입주·분양권 전환 등 실제 사건일 수 있어 목록으로만 확인, 최근 2개월 '적음'은 신고 기간이라 정상",
        "",
    ]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "quality.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"거래 {match['rows']:,}건, trade_key 중복 {dup['dup_trade_key']}, 쌍둥이 {dup['twin_rows']:,}건")
    print(
        f"건수 급변 {len(spikes)}건, 값 범위 위반 {values['trade_key'].nunique():,}건, 가격 이상치 {prices['trade_key'].nunique():,}건"
    )
    print(f"→ reports/quality.md, {FLAGS_OUT.relative_to(ROOT)}")
    return 1 if dup["dup_trade_key"] or len(values) else 0


if __name__ == "__main__":
    sys.exit(main())
