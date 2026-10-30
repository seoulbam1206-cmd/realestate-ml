"""시점 안전 거래 피처표 + 기준선 B0·B1 (PLAN.md MVP 3-3 ~ 3-5).

계약 월 M 마다 기준일 C(M) 까지 공개된 거래(h)만으로 집계를 만들어 M 월 거래에 붙인다.
학습 행과 예측 행이 같은 규칙을 따르므로 피처표는 한 번만 만든다 (모델은 asof.training_rows 로 행만 고름).

- h: 계약일 ≤ C(M), 해제일 ≤ C(M) 인 해제 거래는 뺌 (그 뒤 해제는 당시 정상 거래)
- 단지·동 이력: h 의 최근 12/36개월
- 시군구 시장: h 의 M−2월까지 월별 중위 log ㎡당가 → 3개월 거래 수 가중 지수
- R-ONE: 시군구에 붙인 조사 지수(M−1까지)·실거래가격지수(M−2까지)
- B0 지역 중위가: 법정동 M−4~M−2월 ㎡당 중위가 (5건 미만이면 시군구 → 시도 → 전국)
- B1 단지 중위가: 단지×면적대 최근 12개월 ㎡당 중위가 × 그 뒤 시군구 지수 변화 (없으면 B0)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd

from realestate.features import asof

CATEGORICAL = ["band", "month_of_year", "dealing_gbn", "buyer_gbn", "seller_gbn", "sgg_cd", "sido_cd"]
NUMERIC = [
    "area", "floor", "rel_floor", "age",
    "cx12_med", "cx12_n", "cx12_smooth", "cx36_med", "cx36_n", "cx_last_adj", "cx_last_age_m",
    "bjd12_med", "bjd12_n",
    "sgg_chg1", "sgg_chg3", "sgg_chg12", "sgg_vol_ratio", "sgg_cancel_ratio",
    "rone_s_chg3", "rone_s_chg12", "rone_r_chg3", "rone_r_chg12",
]  # fmt: skip
FEATURES = NUMERIC + CATEGORICAL
SMOOTH_K = 5  # 단지 중위가를 동 중위가 쪽으로 당기는 세기 (건수 5건이면 반반)
B0_MIN_N = 5
AREA_BANDS = (40, 60, 85, 102, 135)  # ≤40 / ≤60 / ≤85 / ≤102 / ≤135 / 135 초과 → 1~6


def month_index(ym: str) -> int:
    return int(ym[:4]) * 12 + int(ym[4:]) - 1


def index_ym(mi: int) -> str:
    return f"{mi // 12}{mi % 12 + 1:02d}"


def load(con: duckdb.DuckDBPyConnection, trades: pd.DataFrame | list[Path]) -> None:
    """정리본(DataFrame 또는 Parquet 파일 목록) → DuckDB 표 t (필요한 칸만, 파생 칸 추가)."""
    if isinstance(trades, pd.DataFrame):
        con.register("src", trades)
        src = "src"
    else:
        src = "read_parquet([{}], union_by_name = true)".format(
            ", ".join(f"'{p.as_posix()}'" for p in trades)
        )
    band = " ".join(f"WHEN exclu_use_ar <= {b} THEN {i}" for i, b in enumerate(AREA_BANDS, 1))
    con.execute(
        f"""
        CREATE OR REPLACE TABLE t AS
        SELECT trade_key, sgg_cd, substr(sgg_cd, 1, 2) AS sido_cd, bjd_cd, complex_key,
               exclu_use_ar::FLOAT AS area, CASE {band} ELSE {len(AREA_BANDS) + 1} END AS band,
               floor::SMALLINT AS floor, build_year::SMALLINT AS build_year,
               deal_date::DATE AS deal_date, (year(deal_date) * 12 + month(deal_date) - 1) AS mi,
               deal_amount::BIGINT AS deal_amount, price_per_m2::DOUBLE AS ppm2, ln(price_per_m2) AS lp,
               cdeal_day::DATE AS cdeal_day, is_canceled::BOOLEAN AS is_canceled,
               dealing_gbn, buyer_gbn, seller_gbn
        FROM {src}
        """
    )


def load_rone(con: duckdb.DuckDBPyConnection, survey: pd.DataFrame, real: pd.DataFrame) -> None:
    """R-ONE 지수 (sgg_cd, ym, value) → 표 rs(조사), rr(실거래가격지수)."""
    for name, df in (("rs", survey), ("rr", real)):
        d = df[["sgg_cd", "ym", "value"]].assign(mi=df["ym"].map(month_index))
        con.register("_r", d)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT sgg_cd, mi, value::DOUBLE AS value FROM _r")
        con.unregister("_r")


def build_month(con: duckdb.DuckDBPyConnection, ym: str) -> tuple[pd.DataFrame, dict]:
    m, c = month_index(ym), asof.cutoff_date(ym)
    cd = f"DATE '{c.isoformat()}'"  # 우리가 만든 날짜라 그대로 넣음 (여러 문장 실행은 매개변수 불가)
    q = lambda sql, params=None: con.execute(sql, params or [])  # noqa: E731
    q(
        "CREATE OR REPLACE TEMP TABLE h AS SELECT * FROM t "
        "WHERE deal_date <= ? AND NOT coalesce(cdeal_day <= ?, false)",
        [c, c],
    )
    q(
        f"""
        CREATE OR REPLACE TEMP TABLE cx12 AS
        SELECT complex_key, band, median(lp) AS med, count(*) AS n, avg(mi) AS mmi
        FROM h WHERE deal_date > {cd} - INTERVAL 365 DAY GROUP BY 1, 2;
        CREATE OR REPLACE TEMP TABLE cx36 AS
        SELECT complex_key, median(lp) AS med, count(*) AS n
        FROM h WHERE deal_date > {cd} - INTERVAL 1095 DAY GROUP BY 1;
        CREATE OR REPLACE TEMP TABLE cxl AS
        SELECT complex_key, band, arg_max(lp, deal_date) AS last_lp, max(mi) AS last_mi FROM h GROUP BY 1, 2;
        CREATE OR REPLACE TEMP TABLE cxf AS SELECT complex_key, max(floor) AS max_floor FROM h GROUP BY 1;
        CREATE OR REPLACE TEMP TABLE bj12 AS
        SELECT bjd_cd, median(lp) AS med, count(*) AS n
        FROM h WHERE deal_date > {cd} - INTERVAL 365 DAY GROUP BY 1;
        CREATE OR REPLACE TEMP TABLE sm AS
        SELECT sgg_cd, mi, median(lp) AS med, count(*) AS n FROM h WHERE mi <= {m - 2} GROUP BY 1, 2;
        CREATE OR REPLACE TEMP TABLE sidx AS
        SELECT sgg_cd, mi, n, sum(n * med) OVER w / sum(n) OVER w AS idx FROM sm
        WINDOW w AS (PARTITION BY sgg_cd ORDER BY mi RANGE BETWEEN 2 PRECEDING AND CURRENT ROW);
        CREATE OR REPLACE TEMP TABLE sf AS
        SELECT sgg_cd,
               max(idx) FILTER (WHERE mi = {m - 2}) AS i2, max(idx) FILTER (WHERE mi = {m - 3}) AS i3,
               max(idx) FILTER (WHERE mi = {m - 5}) AS i5, max(idx) FILTER (WHERE mi = {m - 14}) AS i14,
               sum(n) FILTER (WHERE mi BETWEEN {m - 4} AND {m - 2}) AS v_recent,
               sum(n) FILTER (WHERE mi BETWEEN {m - 16} AND {m - 14}) AS v_prev
        FROM sidx GROUP BY 1;
        CREATE OR REPLACE TEMP TABLE cr AS
        SELECT sgg_cd, avg(CASE WHEN cdeal_day <= {cd} THEN 1.0 ELSE 0.0 END) AS ratio
        FROM t WHERE mi BETWEEN {m - 4} AND {m - 2} AND deal_date <= {cd} GROUP BY 1;
        CREATE OR REPLACE TEMP TABLE b3 AS
        SELECT 'bjd' AS lvl, bjd_cd AS k, median(ppm2) AS med, count(*) AS n
        FROM h WHERE mi BETWEEN {m - 4} AND {m - 2} GROUP BY 2
        UNION ALL SELECT 'sgg', sgg_cd, median(ppm2), count(*) FROM h WHERE mi BETWEEN {m - 4} AND {m - 2} GROUP BY 2
        UNION ALL SELECT 'sido', sido_cd, median(ppm2), count(*) FROM h WHERE mi BETWEEN {m - 4} AND {m - 2} GROUP BY 2
        UNION ALL SELECT 'nation', '', median(ppm2), count(*) FROM h WHERE mi BETWEEN {m - 4} AND {m - 2};
        """,
    )
    df = q(
        f"""
        SELECT r.trade_key, '{ym}' AS ym, r.deal_date, r.deal_amount, r.ppm2, r.lp, r.is_canceled, r.cdeal_day,
               r.complex_key, r.bjd_cd, r.sgg_cd, r.sido_cd,
               r.area, r.band, r.floor, r.floor / nullif(cf.max_floor, 0) AS rel_floor,
               year(r.deal_date) - r.build_year AS age, month(r.deal_date) AS month_of_year,
               r.dealing_gbn, r.buyer_gbn, r.seller_gbn,
               c12.med AS cx12_med, coalesce(c12.n, 0) AS cx12_n,
               CASE WHEN c12.med IS NULL THEN b12.med WHEN b12.med IS NULL THEN c12.med
                    ELSE (c12.n * c12.med + {SMOOTH_K} * b12.med) / (c12.n + {SMOOTH_K}) END AS cx12_smooth,
               c36.med AS cx36_med, coalesce(c36.n, 0) AS cx36_n,
               cl.last_lp + coalesce(sf.i2 - li.idx, 0) AS cx_last_adj, {m} - cl.last_mi AS cx_last_age_m,
               b12.med AS bjd12_med, coalesce(b12.n, 0) AS bjd12_n,
               sf.i2 - sf.i3 AS sgg_chg1, sf.i2 - sf.i5 AS sgg_chg3, sf.i2 - sf.i14 AS sgg_chg12,
               sf.v_recent / nullif(sf.v_prev, 0) AS sgg_vol_ratio, cr.ratio AS sgg_cancel_ratio,
               ln(s1.value / s4.value) AS rone_s_chg3, ln(s1.value / s13.value) AS rone_s_chg12,
               ln(r2.value / r5.value) AS rone_r_chg3, ln(r2.value / r14.value) AS rone_r_chg12,
               coalesce(CASE WHEN bb.n >= {B0_MIN_N} THEN bb.med END, CASE WHEN bs.n >= {B0_MIN_N} THEN bs.med END,
                        CASE WHEN bd.n >= {B0_MIN_N} THEN bd.med END, bn.med) AS b0_ppm2,
               CASE WHEN bb.n >= {B0_MIN_N} THEN 'bjd' WHEN bs.n >= {B0_MIN_N} THEN 'sgg'
                    WHEN bd.n >= {B0_MIN_N} THEN 'sido' ELSE 'nation' END AS b0_level,
               exp(c12.med + coalesce(sf.i2 - bi.idx, 0)) AS b1_cx_ppm2
        FROM t r
        LEFT JOIN cxf cf USING (complex_key)
        LEFT JOIN cx12 c12 ON c12.complex_key = r.complex_key AND c12.band = r.band
        LEFT JOIN cx36 c36 ON c36.complex_key = r.complex_key
        LEFT JOIN cxl cl ON cl.complex_key = r.complex_key AND cl.band = r.band
        LEFT JOIN bj12 b12 ON b12.bjd_cd = r.bjd_cd
        LEFT JOIN sf ON sf.sgg_cd = r.sgg_cd
        LEFT JOIN sidx li ON li.sgg_cd = r.sgg_cd AND li.mi = least(cl.last_mi, {m - 2})
        LEFT JOIN sidx bi ON bi.sgg_cd = r.sgg_cd AND bi.mi = least(round(c12.mmi)::INT, {m - 2})
        LEFT JOIN cr ON cr.sgg_cd = r.sgg_cd
        LEFT JOIN rs s1 ON s1.sgg_cd = r.sgg_cd AND s1.mi = {m - 1}
        LEFT JOIN rs s4 ON s4.sgg_cd = r.sgg_cd AND s4.mi = {m - 4}
        LEFT JOIN rs s13 ON s13.sgg_cd = r.sgg_cd AND s13.mi = {m - 13}
        LEFT JOIN rr r2 ON r2.sgg_cd = r.sgg_cd AND r2.mi = {m - 2}
        LEFT JOIN rr r5 ON r5.sgg_cd = r.sgg_cd AND r5.mi = {m - 5}
        LEFT JOIN rr r14 ON r14.sgg_cd = r.sgg_cd AND r14.mi = {m - 14}
        LEFT JOIN b3 bb ON bb.lvl = 'bjd' AND bb.k = r.bjd_cd
        LEFT JOIN b3 bs ON bs.lvl = 'sgg' AND bs.k = r.sgg_cd
        LEFT JOIN b3 bd ON bd.lvl = 'sido' AND bd.k = r.sido_cd
        LEFT JOIN b3 bn ON bn.lvl = 'nation'
        WHERE r.mi = {m}
        """
    ).df()
    df["b1_level"] = df["b1_cx_ppm2"].notna().map({True: "complex", False: "b0"})
    df["b1_ppm2"] = df["b1_cx_ppm2"].fillna(df["b0_ppm2"])
    df = df.drop(columns="b1_cx_ppm2")

    # 실제로 쓴 원천의 끝 (asof.audit 이 허용 범위와 비교)
    hist_max, cancel_max = q(
        "SELECT max(deal_date), (SELECT max(cdeal_day) FROM t WHERE deal_date <= ? AND cdeal_day <= ?) FROM h",
        [c, c],
    ).fetchone()
    agg_max = q("SELECT max(mi) FROM sm").fetchone()[0]
    s_max = q(f"SELECT max(mi) FROM rs WHERE mi IN ({m - 1}, {m - 4}, {m - 13})").fetchone()[0]
    r_max = q(f"SELECT max(mi) FROM rr WHERE mi IN ({m - 2}, {m - 5}, {m - 14})").fetchone()[0]
    meta = {
        "ym": ym,
        "cutoff": c,
        "rows": len(df),
        "hist_rows": q("SELECT count(*) FROM h").fetchone()[0],
        "hist_max_deal_date": hist_max,
        "hist_max_cancel_day": cancel_max,
        "agg_max_month": None if agg_max is None else index_ym(agg_max),
        "survey_max_month": None if s_max is None else index_ym(s_max),
        "real_max_month": None if r_max is None else index_ym(r_max),
    }
    return df, meta


def iter_months(
    trades: pd.DataFrame | list[Path], months: list[str], survey: pd.DataFrame, real: pd.DataFrame,
    memory_limit: str = "2GB",
) -> Iterator[tuple[pd.DataFrame, dict]]:  # fmt: skip
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    load(con, trades)
    load_rone(con, survey, real)
    try:
        for ym in months:
            yield build_month(con, ym)
    finally:
        con.close()


def build_features(
    trades: pd.DataFrame | list[Path], months: list[str], survey: pd.DataFrame, real: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """작은 자료용 (테스트). 큰 자료는 iter_months 로 달마다 파일에 쓴다."""
    parts, metas = [], []
    for df, meta in iter_months(trades, months, survey, real):
        parts.append(df)
        metas.append(meta)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(metas)


def rone_by_sgg(
    rone_index: pd.DataFrame, region_map: pd.DataFrame, source: str, cover_from: str = "201901"
) -> pd.DataFrame:
    """R-ONE 지수를 시군구에 붙인다 → (sgg_cd, ym, value).

    시군구마다 그 시군구를 포함하는 R-ONE 지역 중 cover_from 이전부터 값이 있는 곳에서 가장 작은 지역
    (시군구 수 기준)을 고른다. 그런 곳이 없으면 가장 작은 지역. 예: 인천 제물포구(2026.6~ 지역)는 인천 시도 지수.
    """
    idx = rone_index[rone_index["source"] == source].assign(ym=lambda d: d["ym"].dt.strftime("%Y%m"))
    first = idx.groupby("region_key")["ym"].min()
    rmap = region_map.dropna(subset=["sgg_cd"])
    rmap = rmap[rmap["region_key"].isin(first.index)].copy()
    rmap["size"] = rmap.groupby("region_key")["sgg_cd"].transform("size")
    rmap["late"] = rmap["region_key"].map(first) > cover_from
    pick = rmap.sort_values(["sgg_cd", "late", "size", "region_key"]).drop_duplicates("sgg_cd")
    out = pick[["sgg_cd", "region_key"]].merge(idx[["region_key", "ym", "index_value"]], on="region_key")
    return out.rename(columns={"index_value": "value"})[["sgg_cd", "ym", "value", "region_key"]]
