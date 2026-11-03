"""호가 비교용 시세 모델 (PLAN.md MVP 4): 기준일 D = 2026.9.30 모델 1개 + 가상 거래 행의 피처 계산.

- 설정은 고정 c05 (config/lgbm.json) 그대로. 학습 절차도 walk-forward 와 같음 (lgbm.fit)
- 예측 시점 T = 2026.10 (anchor_for("202610") == 2026-09-30, walk-forward 규칙과 같음).
  T = 2026.11 이면 R-ONE 조사 지수 2026.10·실거래가격지수 2026.9 가 필요한데 아직 발표 전이라 피처 4개가 빔
  → 피처가 모두 채워지는 마지막 달 2026.10 (정보 기준일 C(T) = 2026.8.31)
- 피처: 입력(단지·면적·층)으로 만든 가상 거래 행을 T 월 거래로 표 t 에 넣고 trade_features.build_month 를
  그대로 돌림 → 학습과 같은 SQL. 가상 행은 C(T) 이후 계약이라 이력(h)에는 들어가지 않음
- 거래 구분 범주(dealing_gbn·buyer_gbn·seller_gbn)는 가장 흔한 값으로 고정
- 오차 범위: 평가 기간(2024.1~2025.9) 실제 오차 ln(실거래/예측) 의 10%·90% 분위 (단지·면적대 최근 12개월 거래 수 구간별)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from realestate.config import ROOT

if TYPE_CHECKING:
    import lightgbm as lgb
# lightgbm·학습 코드는 함수 안에서 불러옴: 화면(Streamlit)은 ID·층 구간·판정만 쓰므로 lightgbm 없이 import 됨

LGBM_CONFIG = ROOT / "config" / "lgbm.json"
ANCHOR = date(2026, 9, 30)
TARGET_YM = "202610"
SYNTH_PREFIX = "synth:"
DEFAULT_GBN = {"dealing_gbn": "중개거래", "buyer_gbn": "개인", "seller_gbn": "개인"}
MODEL_NAME = "apt_price_lgbm"
ERROR_BAND_PATH = ROOT / "config" / "price_error_band.json"

# 최근 12개월 거래 수(cx12_n) 구간 → 이름 (오차 범위를 나누는 기준)
HISTORY_GROUPS = [(0, 0, "0건"), (1, 4, "1~4건"), (5, 19, "5~19건"), (20, 10**9, "20건 이상")]
FLOOR_BUCKETS = [  # 적정가 표의 층 구간 (대표 층)
    ("1~3층", 1, 3, 2),
    ("4~7층", 4, 7, 5),
    ("8~12층", 8, 12, 10),
    ("13~20층", 13, 20, 16),
    ("21층 이상", 21, 999, 25),
]


def model_dir() -> Path:
    cfg = json.loads(LGBM_CONFIG.read_text(encoding="utf-8"))
    return ROOT / "data" / "models" / "lgbm" / cfg["params_hash"] / "serve"


def model_paths() -> tuple[Path, Path]:
    d = model_dir()
    return d / f"D={ANCHOR:%Y%m%d}.txt", d / f"D={ANCHOR:%Y%m%d}_meta.json"


def basis_text(ts: pd.Timestamp | None = None) -> str:
    """'2026년 10월 기준 적정가' (화면·API 공통 표기)."""
    ts = ts if ts is not None else pd.Timestamp(f"{TARGET_YM[:4]}-{TARGET_YM[4:]}-01")
    return f"{ts.year}년 {ts.month}월 기준 적정가"


def ascii_tempdir() -> None:
    """LightGBM 은 한글이 든 경로에 모델 파일을 쓰지 못함 (Windows 사용자 임시 폴더가 그런 경우)
    → MLflow 가 쓰는 임시 폴더를 프로젝트 안 mlruns/_tmp (git 제외) 로."""
    import tempfile

    d = ROOT / "mlruns" / "_tmp"
    d.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(d)


def public_complex_id(complex_key: str) -> str:
    """공개 단지 ID: 내부 단지 해시를 한 번 더 해시 (내부 키는 공개하지 않음, 다시 만들어도 같음)."""
    return "c_" + hashlib.sha1(f"public:{complex_key}".encode()).hexdigest()[:10]


def history_group(cx12_n: float) -> str:
    n = 0 if pd.isna(cx12_n) else int(cx12_n)
    for lo, hi, name in HISTORY_GROUPS:
        if lo <= n <= hi:
            return name
    raise ValueError(n)


def floor_bucket(floor: int) -> str:
    for name, lo, hi, _ in FLOOR_BUCKETS:
        if lo <= floor <= hi:
            return name
    return FLOOR_BUCKETS[0][0]  # 지하·0층은 저층으로


@dataclass
class ServeModel:
    booster: lgb.Booster
    cats: dict[str, list]
    meta: dict

    @classmethod
    def load(cls, source: str = "file") -> ServeModel:
        """source='file': data/models/.../serve 의 모델 파일. 'registry': MLflow Registry apt_price_lgbm
        (meta 에 적힌 등록 버전). 범주 목록·기록은 둘 다 meta 파일에서."""
        txt, meta_path = model_paths()
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} 가 없습니다. 먼저 scripts/build_fair_price.py train")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if source == "registry":
            import mlflow
            import mlflow.lightgbm

            from realestate.models import tracking

            mlflow.set_tracking_uri(tracking.TRACKING_URI)
            ascii_tempdir()
            booster = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{meta['registry_version']}")
            meta = {**meta, "loaded_from": f"models:/{MODEL_NAME}/{meta['registry_version']}"}
        else:
            import lightgbm as lgb

            booster = lgb.Booster(model_file=str(txt))
            meta = {**meta, "loaded_from": str(txt.relative_to(ROOT).as_posix())}
        return cls(booster, meta["cats"], meta)

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        """거래가 (만 원)."""
        from realestate.models import lgbm

        return lgbm.predict_price(self.booster, feats, self.cats)


def train_and_save() -> dict:
    from realestate.models import lgbm

    cfg = json.loads(LGBM_CONFIG.read_text(encoding="utf-8"))
    assert lgbm.anchor_for(TARGET_YM) == ANCHOR
    train = lgbm.training_frame(ANCHOR)
    assert train["deal_date"].max() <= pd.Timestamp(ANCHOR)
    model, cats, info = lgbm.fit(train, ANCHOR, cfg["params"])
    txt, meta_path = model_paths()
    txt.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(txt))
    meta = {
        "anchor": str(ANCHOR),
        "target_ym": TARGET_YM,
        "params_hash": cfg["params_hash"],
        "cand": cfg["cand"],
        "feature_set": lgbm.feature_set_hash(),
        "train_last_deal_date": str(train["deal_date"].max().date()),
        **info,
        "cats": cats,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8")
    return meta


# ---------------------------------------------------------------- 피처 (가상 거래 행)


def synth_rows(req: pd.DataFrame) -> pd.DataFrame:
    """요청(complex_key, sgg_cd, bjd_cd, build_year, area, floor, req_id) → 정리본 모양의 가상 거래 행 (T 월 1일 계약)."""
    d = pd.Timestamp(f"{TARGET_YM[:4]}-{TARGET_YM[4:]}-01")
    return pd.DataFrame(
        {
            "trade_key": SYNTH_PREFIX + req["req_id"].astype(str),
            "sgg_cd": req["sgg_cd"].astype(str),
            "bjd_cd": req["bjd_cd"].astype(str),
            "complex_key": req["complex_key"].astype(str),
            "exclu_use_ar": req["area"].astype(float),
            "floor": req["floor"].astype(int),
            "build_year": req["build_year"].astype("Int64"),
            "deal_date": d,
            "deal_amount": pd.array([None] * len(req), dtype="Int64"),
            "price_per_m2": np.nan,
            "cdeal_day": pd.NaT,
            "is_canceled": False,
            **{k: v for k, v in DEFAULT_GBN.items()},
        }
    )


def add_synth(con, req: pd.DataFrame) -> None:
    """DuckDB 표 t (trade_features.load 로 만든 것) 에 가상 행 추가."""
    from realestate.features import trade_features as tf

    s = synth_rows(req)
    con.register("_synth", s)
    band = " ".join(f"WHEN exclu_use_ar <= {b} THEN {i}" for i, b in enumerate(tf.AREA_BANDS, 1))
    con.execute(
        f"""
        INSERT INTO t BY NAME
        SELECT trade_key, sgg_cd, substr(sgg_cd, 1, 2) AS sido_cd, bjd_cd, complex_key,
               exclu_use_ar::FLOAT AS area, CASE {band} ELSE {len(tf.AREA_BANDS) + 1} END AS band,
               floor::SMALLINT AS floor, build_year::SMALLINT AS build_year,
               deal_date::DATE AS deal_date, (year(deal_date) * 12 + month(deal_date) - 1) AS mi,
               NULL::BIGINT AS deal_amount, NULL::DOUBLE AS ppm2, NULL::DOUBLE AS lp,
               NULL::DATE AS cdeal_day, false AS is_canceled, dealing_gbn, buyer_gbn, seller_gbn
        FROM _synth
        """
    )
    con.unregister("_synth")


def synth_features(con, req: pd.DataFrame) -> pd.DataFrame:
    """가상 행 피처 (req_id 순서 유지). 실행 뒤 가상 행은 t 에서 지움."""
    from realestate.features import trade_features as tf

    add_synth(con, req)
    try:
        df, _ = tf.build_month(con, TARGET_YM)
    finally:
        con.execute(f"DELETE FROM t WHERE trade_key LIKE '{SYNTH_PREFIX}%'")
    df = df[df["trade_key"].str.startswith(SYNTH_PREFIX)].copy()
    df["req_id"] = df["trade_key"].str[len(SYNTH_PREFIX) :]
    order = req["req_id"].astype(str).tolist()
    return df.set_index("req_id").loc[order].reset_index()


def open_history(memory_limit: str = "3GB"):
    """정리본 전체 + R-ONE 을 DuckDB 에 올린 연결 (피처 계산용, build_features.py 와 같은 원천)."""
    import duckdb

    from realestate.config import BJD_RAW, REFERENCE_CLEAN, RONE_CLEAN
    from realestate.features import trade_features as tf
    from realestate.ingest import bjd, trades

    src = trades.month_sources(bjd.lawd_codes(bjd.load(BJD_RAW)))
    files = [p for f in src["files"] for p in f]
    rone = pd.read_parquet(RONE_CLEAN / "rone_index.parquet")
    rmap = pd.read_parquet(REFERENCE_CLEAN / "region_map.parquet")
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    tf.load(con, files)
    tf.load_rone(con, tf.rone_by_sgg(rone, rmap, "survey"), tf.rone_by_sgg(rone, rmap, "real"))
    return con


def _mode_sql(col: str) -> str:
    """단지별 최빈값 (같은 횟수면 작은 값) — 프로세스마다 같은 결과가 나오도록 순서를 고정."""
    return f"""
        SELECT complex_key, {col} FROM (
            SELECT complex_key, {col}, row_number() OVER (
                PARTITION BY complex_key ORDER BY count(*) DESC, {col}) AS rn
            FROM h_serve WHERE {col} IS NOT NULL GROUP BY 1, 2
        ) WHERE rn = 1"""


def history_view(con) -> None:
    """임시 뷰 h_serve: 정보 기준일 C(T) 까지 공개된 정상 거래 (build_month 의 h 와 같은 조건, 가상 행 제외)."""
    from realestate.features import asof

    c = asof.cutoff_date(TARGET_YM).isoformat()
    con.execute(
        f"""CREATE OR REPLACE TEMP VIEW h_serve AS SELECT * FROM t
        WHERE deal_date <= DATE '{c}' AND NOT coalesce(cdeal_day <= DATE '{c}', false)
          AND trade_key NOT LIKE '{SYNTH_PREFIX}%'"""
    )


def complex_index(con) -> pd.DataFrame:
    """C(T) 까지 이력이 있는 단지: complex_id, complex_key, sgg_cd, bjd_cd(최빈), build_year(최빈), max_floor, 이름."""
    history_view(con)
    df = con.execute(
        f"""
        WITH b AS ({_mode_sql("bjd_cd")}), y AS ({_mode_sql("build_year")}),
        base AS (SELECT complex_key, min(sgg_cd) AS sgg_cd, max(floor) AS max_floor FROM h_serve GROUP BY 1)
        SELECT base.complex_key, base.sgg_cd, b.bjd_cd, y.build_year, base.max_floor
        FROM base LEFT JOIN b USING (complex_key) LEFT JOIN y USING (complex_key)
        """
    ).df()
    df = df.merge(complex_names(df["complex_key"]), on="complex_key", how="left", validate="1:1")
    df["complex_id"] = df["complex_key"].map(public_complex_id)
    assert df["complex_id"].is_unique, "공개 ID 충돌"
    return df


def complex_names(keys: pd.Series) -> pd.DataFrame:
    """complex_key → (apt_nm, umd_nm 법정동 이름). complex_map·법정동 코드표 기준."""
    from realestate.config import REFERENCE_CLEAN

    cmap = pd.read_parquet(
        REFERENCE_CLEAN / "complex_map.parquet", columns=["complex_key", "bjd_cd", "apt_nm"]
    )
    bjd = pd.read_parquet(REFERENCE_CLEAN / "bjd_code.parquet", columns=["bjd_cd", "full_nm"])
    bjd = bjd.drop_duplicates("bjd_cd").set_index("bjd_cd")["full_nm"].str.split().str[-1]
    out = pd.DataFrame({"complex_key": keys.drop_duplicates()}).merge(cmap, on="complex_key", how="left")
    out["umd_nm"] = out["bjd_cd"].map(bjd).fillna("")
    return out[["complex_key", "apt_nm", "umd_nm"]]


# ---------------------------------------------------------------- 오차 범위


def load_error_band() -> dict:
    return json.loads(ERROR_BAND_PATH.read_text(encoding="utf-8"))


FEW_HISTORY_NOTE = "이력이 적어 오차가 큼"
ASK_NOTE = (
    "단지·개별 단위는 검증하지 않은 참고 정보입니다. "
    "호가는 보통 실거래가보다 높게 나오므로 '모델 대비 높음'이 흔합니다. "
    "모델보다 낮다는 것이 이후 더 오른다는 근거는 없습니다(사후 분석 참고)."
)


def price_range(fair: float, group: str, bands: dict) -> tuple[float, float]:
    """80% 범위 (만 원): 적정가 × exp(q10) ~ × exp(q90). 잔차 = ln(실거래 / 예측) 의 평가 기간 분위."""
    b = bands[group]
    return fair * float(np.exp(b["q10"])), fair * float(np.exp(b["q90"]))


def judge(ask: float, fair: float, low: float, high: float, group: str) -> dict:
    """호가 판정. 80% 범위 안이면 '모델과 비슷', 위면 '모델 대비 높음', 아래면 '모델 대비 낮음'.

    최근 12개월 거래 0건 구간은 범위가 넓어 판정 옆에 '이력이 적어 오차가 큼' 을 붙임.
    """
    if ask > high:
        label = "모델 대비 높음"
    elif ask < low:
        label = "모델 대비 낮음"
    else:
        label = "모델과 비슷"
    return {
        "ask_gap_pct": round((ask / fair - 1) * 100, 1),
        "judgement": label,
        "judgement_note": FEW_HISTORY_NOTE if group == "0건" else None,
    }
