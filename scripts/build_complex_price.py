"""단지 시세 화면용 집계표 → data/public/complex_price.parquet (개별 거래 행 없음, 등급 없음, 정보용).

| 열 | 내용 |
|---|---|
| row_id | 행 번호 (같은 동에 이름이 같은 단지가 있어 키로 씀. 단지 식별 해시는 올리지 않음) |
| sgg_cd, umd_nm, apt_nm | 시군구 코드, 법정동 이름, 단지명 (complex_map) |
| build_year | 건축년도 (창 안 거래의 최빈값, 같은 동·같은 이름 단지 구분용) |
| trade_cnt | 최근 12개월 정상 거래 수 (지금 기준 해제 아님) |
| median_ppm2 | 실거래 ㎡당가 중위값 (만 원/㎡) |
| model_ppm2 | 같은 거래들의 모델 적정가 ㎡당가 중위값 (만 원/㎡) |
| gap_pct | 거래마다 (실거래가 ÷ 모델 적정가 − 1) 의 중위값 × 100. +면 모델보다 높게 거래됨 |
| window_from, window_to | 집계 계약월 (첫 달, 끝 달) |

- 모델 적정가 = 시세 모델 LightGBM 고정 설정(config/lgbm.json) 의 walk-forward 예측 (그 거래를 학습에 안 쓴 모델).
  이미 저장된 예측만 읽음 (새로 학습하지 않음)
- 12개월 창의 끝 = 정상 거래 전부에 예측이 있는 마지막 계약월. 창 안에 예측 없는 정상 거래가 있으면 멈춤
- 거래 < 5건 단지는 가격 칸(median_ppm2·model_ppm2·gap_pct) 을 비움 (한두 건이면 개별 거래가가 드러남)
- 단지 단위는 검증하지 않은 참고 정보. 매매 점수·등급과 무관

실행: uv run python scripts/build_complex_price.py
"""

from __future__ import annotations

import json
import sys

import duckdb
import numpy as np
import pandas as pd

from realestate.config import APT_TRADE_CLEAN, REFERENCE_CLEAN, ROOT
from realestate.models.periods import FEATURE_DIR

OUT = ROOT / "data" / "public" / "complex_price.parquet"
LGBM_CONFIG = ROOT / "config" / "lgbm.json"
PRICE_MIN_N = 5
MONTHS = 12


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = json.loads(LGBM_CONFIG.read_text(encoding="utf-8"))
    pred_glob = (ROOT / "data" / "models" / "lgbm" / cfg["params_hash"] / "D=*[0-9].parquet").as_posix()
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'")
    con.execute(f"CREATE VIEW pred AS SELECT trade_key, pred_lgbm FROM read_parquet('{pred_glob}')")
    assert con.execute("SELECT count(*) - count(DISTINCT trade_key) FROM pred").fetchone()[0] == 0, (
        "예측 중복"
    )
    con.execute(
        f"""CREATE TABLE t AS
        SELECT f.trade_key, f.ym, f.complex_key, f.sgg_cd, f.ppm2, f.area, f.deal_amount, p.pred_lgbm
        FROM read_parquet('{(FEATURE_DIR / "ym=*.parquet").as_posix()}', union_by_name = true) f
        LEFT JOIN pred p USING (trade_key)
        WHERE NOT f.is_canceled"""
    )
    cover = con.sql("SELECT ym, count(*) n, count(pred_lgbm) k FROM t GROUP BY 1 ORDER BY 1").df()
    full = cover[cover["n"] == cover["k"]]["ym"]
    end = pd.Period(f"{full.max()[:4]}-{full.max()[4:]}", freq="M")
    window = [f"{p.year}{p.month:02d}" for p in pd.period_range(end - (MONTHS - 1), end, freq="M")]
    missing = cover[cover["ym"].isin(window) & (cover["n"] != cover["k"])]
    assert missing.empty, f"창 안에 예측 없는 정상 거래:\n{missing}"
    later = cover.loc[cover["ym"] > window[-1], "ym"]
    print(f"창: {window[0]} ~ {window[-1]} (그 뒤 예측 없는 달: {', '.join(later) or '없음'})")

    df = con.sql(
        f"""
        SELECT complex_key, any_value(sgg_cd) AS sgg_cd, count(*) AS trade_cnt,
               median(ppm2) AS median_ppm2,
               median(pred_lgbm / area) AS model_ppm2,
               median(deal_amount / pred_lgbm - 1) * 100 AS gap_pct
        FROM t WHERE ym IN ({", ".join(f"'{m}'" for m in window)})
        GROUP BY 1
        """
    ).df()
    # 건축년도: 창 안 정상 거래의 build_year 최빈값 (같은 동에 이름이 같은 단지를 구분하는 데 씀)
    by = con.sql(
        f"""
        SELECT complex_key, mode(TRY_CAST(build_year AS INTEGER)) AS build_year
        FROM read_parquet('{(APT_TRADE_CLEAN / "**" / "*.parquet").as_posix()}', union_by_name = true)
        WHERE NOT is_canceled AND strftime(deal_date, '%Y%m') IN ({", ".join(f"'{m}'" for m in window)})
        GROUP BY 1
        """
    ).df()
    df = df.merge(by, on="complex_key", how="left", validate="1:1")
    cmap = pd.read_parquet(
        REFERENCE_CLEAN / "complex_map.parquet", columns=["complex_key", "bjd_cd", "apt_nm"]
    )
    bjd = pd.read_parquet(REFERENCE_CLEAN / "bjd_code.parquet", columns=["bjd_cd", "full_nm"])
    bjd = bjd.drop_duplicates("bjd_cd").set_index("bjd_cd")["full_nm"].str.split().str[-1]
    df = df.merge(cmap, on="complex_key", how="left", validate="1:1")
    assert df["apt_nm"].notna().all(), "단지명이 없는 단지"
    df["umd_nm"] = df["bjd_cd"].map(bjd).fillna("")
    few = df["trade_cnt"] < PRICE_MIN_N
    df.loc[few, ["median_ppm2", "model_ppm2", "gap_pct"]] = np.nan
    first, last = (pd.Timestamp(f"{m[:4]}-{m[4:]}-01").date() for m in (window[0], window[-1]))
    out = (
        df.assign(window_from=first, window_to=last)[
            [
                "sgg_cd",
                "umd_nm",
                "apt_nm",
                "build_year",
                "trade_cnt",
                "median_ppm2",
                "model_ppm2",
                "gap_pct",
                "window_from",
                "window_to",
            ]
        ]  # fmt: skip
        .sort_values(["sgg_cd", "umd_nm", "apt_nm", "build_year"])
        .reset_index(drop=True)
    )
    out["trade_cnt"] = out["trade_cnt"].astype("int32")
    out["build_year"] = out["build_year"].astype("Int16")
    print(
        f"건축년도 없는 단지 {out['build_year'].isna().sum():,} · (시군구·동·이름·건축년도) 중복 "
        f"{out.duplicated(['sgg_cd', 'umd_nm', 'apt_nm', 'build_year']).sum():,}"
    )
    out.insert(
        0, "row_id", np.arange(1, len(out) + 1, dtype="int32")
    )  # 같은 동에 같은 이름 단지가 있어 키로 씀
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(
        f"complex_price: {len(out):,}행 × {out.shape[1]}열 (가격 있는 단지 {(~few).sum():,}, 5건 미만 {few.sum():,})"
    )
    print("  " + ", ".join(f"{c}({t})" for c, t in out.dtypes.astype(str).items()))
    print(out["gap_pct"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
