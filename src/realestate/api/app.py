"""로컬 전용 API (PLAN.md MVP 4): 가격 예측 · 지역 등급 조회. 공개 배포하지 않음.

실행: uv run uvicorn realestate.api.app:app --port 8000   → 문서 http://127.0.0.1:8000/docs

- GET  /health            모델·데이터 기준
- POST /predict           단지 ID · 전용면적 · 층 (· 호가) → 모델 적정가, 80% 범위, 판정, 지역 등급
- GET  /region/{sgg_cd}   시군구 지역 등급 · 점수 · 4요소 · 한 줄 사유

/predict 는 적정가 표(화면)와 달리 입력한 면적·층 그대로 계산한다. 같은 모델·같은 피처 SQL 이라 표의 칸과 같은
면적·층을 넣으면 표 값과 같다 (tests/test_api.py). 첫 /predict 때 거래 이력을 DuckDB 에 올리느라 약 7초, 그 뒤 요청마다 약 1.5초 (2026-11-03 실측).
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

from realestate.config import ROOT
from realestate.models import serve
from realestate.score import explain

PUBLIC = ROOT / "data" / "public"

app = FastAPI(
    title="realestate-ml 로컬 API",
    description="공공 자료를 가공한 참고 정보이며 투자 조언이 아닙니다. " + serve.ASK_NOTE,
    version="0.4.0",
)


# ---------------------------------------------------------------- 자료 (처음 쓸 때 한 번 읽음)


@lru_cache(maxsize=1)
def region_table() -> pd.DataFrame:
    score = pd.read_parquet(PUBLIC / "region_score.parquet")
    names = pd.read_parquet(PUBLIC / "region_name.parquet")
    return score.merge(names, on="sgg_cd", how="left").set_index("sgg_cd")


@lru_cache(maxsize=1)
def error_bands() -> dict:
    return serve.load_error_band()["groups"]


class Predictor:
    """모델 + 거래 이력(DuckDB) + 단지 목록. DuckDB 연결은 한 번에 한 요청만 쓰도록 잠금."""

    def __init__(self) -> None:
        # MLflow Registry 에 등록돼 있으면 거기서, 아니면 모델 파일에서
        meta = serve.model_paths()[1]
        registered = meta.exists() and '"registry_version"' in meta.read_text(encoding="utf-8")
        self.model = serve.ServeModel.load(source="registry" if registered else "file")
        self.con = serve.open_history()
        self.complexes = serve.complex_index(self.con).set_index("complex_id")
        self.lock = threading.Lock()

    def features(self, req: pd.DataFrame) -> pd.DataFrame:
        with self.lock:
            return serve.synth_features(self.con, req)


_predictor: Predictor | None = None
_predictor_lock = threading.Lock()


def predictor() -> Predictor:
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            _predictor = Predictor()
    return _predictor


# ---------------------------------------------------------------- 입력·출력


class PredictIn(BaseModel):
    complex_id: str = Field(pattern=r"^c_[0-9a-f]{10}$", examples=["c_8f3a1b2c4d"])
    area_m2: float = Field(gt=10, lt=400, description="전용면적 (㎡)", examples=[84.97])
    floor: int = Field(ge=1, le=99, examples=[12])
    ask_price_manwon: int | None = Field(default=None, gt=0, description="호가 (만 원). 없으면 적정가만")


def region_out(sgg_cd: str) -> dict | None:
    t = region_table()
    if sgg_cd not in t.index:
        return None
    r = t.loc[sgg_cd]
    flags = [f for f in str(r["rule_flags"] or "").split(",") if f]
    return {
        "label": explain.REGION_LABEL,
        "sgg_cd": sgg_cd,
        "sido_nm": r["sido_nm"],
        "sgg_nm": r["sgg_nm"],
        "asof_month": pd.Timestamp(r["asof_ym"]).strftime("%Y-%m"),
        "grade": r["grade"],
        "score": None if pd.isna(r["score"]) else round(float(r["score"]), 1),
        "reason": explain.reason(r),
        "factors": {
            explain.FACTORS[k]: None if pd.isna(r[k]) else round(float(r[k]), 1) for k in explain.FACTORS
        },
        "rules": [explain.RULE_LABEL.get(f, f) for f in flags],
        "notice": explain.REGION_NOTE,
    }


@app.get("/health")
def health() -> dict:
    txt, meta_path = serve.model_paths()
    meta = {}
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "model": {
            "name": serve.MODEL_NAME,
            "file_exists": txt.exists(),
            "anchor": meta.get("anchor"),
            "target_ym": f"{serve.TARGET_YM[:4]}-{serve.TARGET_YM[4:]}",
            "price_basis": serve.basis_text(),
            "params_hash": meta.get("params_hash"),
            "registry_version": meta.get("registry_version"),
        },
        "history_loaded": _predictor is not None,
        "region_asof_month": pd.Timestamp(region_table()["asof_ym"].max()).strftime("%Y-%m"),
    }


@app.get("/region/{sgg_cd}")
def region(sgg_cd: Annotated[str, Path(pattern=r"^\d{5}$", examples=["11680"])]) -> dict:
    out = region_out(sgg_cd)
    if out is None:
        raise HTTPException(404, f"시군구 {sgg_cd} 의 지역 점수가 없습니다")
    return out


@app.post("/predict")
def predict(body: PredictIn) -> dict:
    p = predictor()
    if body.complex_id not in p.complexes.index:
        raise HTTPException(422, f"단지 {body.complex_id} 가 없습니다 (적정가 표의 complex_id 를 쓰세요)")
    cx = p.complexes.loc[body.complex_id]
    req = pd.DataFrame(
        [
            {
                "req_id": 0,
                "complex_key": cx["complex_key"],
                "sgg_cd": cx["sgg_cd"],
                "bjd_cd": cx["bjd_cd"],
                "build_year": cx["build_year"],
                "area": body.area_m2,
                "floor": body.floor,
            }
        ]
    )
    feats = p.features(req)
    fair = float(p.model.predict(feats)[0])
    group = serve.history_group(feats["cx12_n"].iloc[0])
    low, high = serve.price_range(fair, group, error_bands())
    warnings = []
    if pd.notna(cx["max_floor"]) and body.floor > cx["max_floor"]:
        warnings.append(f"입력한 층이 이 단지에서 거래된 최고층({int(cx['max_floor'])}층)보다 높음")
    out = {
        "model": {
            "name": serve.MODEL_NAME,
            "params_hash": p.model.meta["params_hash"],
            "anchor": p.model.meta["anchor"],
            "target_ym": f"{serve.TARGET_YM[:4]}-{serve.TARGET_YM[4:]}",
            "registry_version": p.model.meta.get("registry_version"),
            "loaded_from": p.model.meta["loaded_from"],
        },
        "complex": {
            "complex_id": body.complex_id,
            "sgg_cd": cx["sgg_cd"],
            "umd_nm": cx["umd_nm"],
            "apt_nm": cx["apt_nm"],
            "build_year": None if pd.isna(cx["build_year"]) else int(cx["build_year"]),
        },
        "input": {"area_m2": body.area_m2, "floor": body.floor},
        "price_basis": serve.basis_text(),
        "fair_price_manwon": round(fair),
        "fair_ppm2_manwon": round(fair / body.area_m2, 1),
        "range_80": {"low_manwon": round(low), "high_manwon": round(high), "history_group": group},
        "ask_price_manwon": body.ask_price_manwon,
        "ask_gap_pct": None,
        "judgement": None,
        "judgement_note": None,
        "region": region_out(cx["sgg_cd"]),
        "warnings": warnings,
        "notice": serve.ASK_NOTE,
    }
    if body.ask_price_manwon:
        out |= serve.judge(body.ask_price_manwon, fair, low, high, group)
    return out
