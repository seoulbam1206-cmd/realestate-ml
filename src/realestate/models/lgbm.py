"""LightGBM 시세 모델 (PLAN.md MVP 3-5). 타깃 log ㎡당가(lp), 예측 거래가 = exp(예측) × 전용면적 (만원).

- 기준일 D(분기 말일)에 학습한 모델을 그 다음 분기 3개월(T)에 그대로 쓴다 (MVP 3-2)
- 학습 행: 2021.1 ~, asof.training_rows(D) (계약일 ≤ D − 30일, 그때까지 공개된 해제는 뺌)
- 조기 종료: 학습 행 중 마지막 완결 2개월(+ 그 뒤 일부 날짜)을 검증으로 떼어 트리 수를 정하고,
  같은 트리 수로 학습 행 전부에 다시 학습 (튜닝·평가가 같은 절차)
"""

from __future__ import annotations

import hashlib
import time
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from realestate.features import asof
from realestate.features.trade_features import CATEGORICAL, FEATURES, NUMERIC
from realestate.models.periods import FEATURE_DIR

TRAIN_FROM = "202101"  # 2020 은 워밍업 (피처 이력만)
VALID_MONTHS = 2
NUM_THREADS = 4
MAX_ROUNDS = 5000
EARLY_STOP = 100
BASE_PARAMS = {
    "learning_rate": 0.1,  # 0.05 는 실측 트리 4,433개·644초 (2026-10-30) → 튜닝 시간 때문에 0.1
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "objective": "l2",
    "metric": "l1",  # log 오차의 절댓값 ≈ 오차율 → MAPE 와 같은 방향
    "num_threads": NUM_THREADS,
    "seed": 0,
    "verbosity": -1,
}
LOAD_COLS = ["ym", "deal_date", "cdeal_day", "is_canceled", "lp", "deal_amount", *FEATURES]


def feature_set_hash() -> str:
    return hashlib.sha1(",".join(FEATURES).encode()).hexdigest()[:10]


def params_hash(params: dict) -> str:
    return hashlib.sha1(repr(sorted(params.items())).encode()).hexdigest()[:10]


def anchor_for(ym: str) -> date:
    """T 월 예측에 쓰는 모델의 기준일 D = T 가 속한 분기의 직전 분기 말일."""
    first = pd.Period(f"{ym[:4]}-{ym[4:]}", freq="Q").start_time.date()
    return first - timedelta(days=1)


def ym_of(d: date) -> str:
    return f"{d.year}{d.month:02d}"


def load_months(first: str, last: str, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(p for p in FEATURE_DIR.glob("ym=*.parquet") if first <= p.stem[3:] <= last)
    return pd.concat(
        (pd.read_parquet(p, columns=list(dict.fromkeys(columns or LOAD_COLS))) for p in files),
        ignore_index=True,
    )


def training_frame(d: date) -> pd.DataFrame:
    return asof.training_rows(load_months(TRAIN_FROM, ym_of(d)), d).reset_index(drop=True)


def valid_first_ym(d: date) -> str:
    """검증 첫 달: 기준일 D 에 신고가 끝난 마지막 완결 달에서 VALID_MONTHS − 1 달 앞."""
    c = d - timedelta(days=asof.REPORT_DAYS)
    last_full = pd.Period(c + timedelta(days=1), freq="M") - 1
    p = last_full - (VALID_MONTHS - 1)
    return f"{p.year}{p.month:02d}"


def categories(train: pd.DataFrame) -> dict[str, list]:
    return {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL}


def matrix(df: pd.DataFrame, cats: dict[str, list]) -> pd.DataFrame:
    """피처 행렬: 수치 float32, 범주는 학습 때 본 값만 (처음 보는 값은 결측)."""
    x = df[NUMERIC].astype("float32")
    for c in CATEGORICAL:
        x[c] = pd.Categorical(df[c], categories=cats[c])
    return x


def fit(train: pd.DataFrame, d: date, params: dict) -> tuple[lgb.Booster, dict, dict]:
    """조기 종료로 트리 수 결정 → 학습 행 전부로 다시 학습. (모델, 범주 목록, 기록)"""
    p = {**BASE_PARAMS, **params}
    cats = categories(train)
    is_valid = (train["ym"] >= valid_first_ym(d)).to_numpy()
    x, y = matrix(train, cats), train["lp"].to_numpy("float32")
    t0 = time.perf_counter()
    dtr = lgb.Dataset(x[~is_valid], y[~is_valid], categorical_feature=CATEGORICAL, free_raw_data=True)
    dva = lgb.Dataset(x[is_valid], y[is_valid], reference=dtr, free_raw_data=True)
    es = lgb.train(
        p, dtr, MAX_ROUNDS, valid_sets=[dva],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )  # fmt: skip
    best, valid_l1 = es.best_iteration, es.best_score["valid_0"]["l1"]
    t_es = time.perf_counter() - t0
    del es, dtr, dva
    dall = lgb.Dataset(x, y, categorical_feature=CATEGORICAL, free_raw_data=True)
    model = lgb.train(p, dall, best)
    info = {
        "n_train": int(len(train)),
        "n_valid": int(is_valid.sum()),
        "best_iter": int(best),
        "valid_l1": float(valid_l1),
        "sec_early_stop": round(t_es, 1),
        "sec_total": round(time.perf_counter() - t0, 1),
    }
    return model, cats, info


def predict_price(model: lgb.Booster, df: pd.DataFrame, cats: dict[str, list]) -> np.ndarray:
    return np.exp(model.predict(matrix(df, cats), num_threads=NUM_THREADS)) * df["area"].to_numpy(float)
