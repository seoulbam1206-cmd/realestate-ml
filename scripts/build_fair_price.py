"""호가 비교용 적정가 (PLAN.md MVP 4). 기준일 D = 2026.9.30 모델 → 단지 × 면적 타입 × 층 구간 적정가 표.

명령
  train     고정 설정 c05 로 기준일 2026.9.30 모델 1회 학습 → data/models/lgbm/<hash>/serve/ (모델 텍스트 + 범주 목록)
  register  학습한 모델을 MLflow Registry 에 apt_price_lgbm 으로 등록 (실험 price_serve)
  band      평가 기간(2024.1~2025.9) walk-forward 예측의 실제 오차 분위 → config/price_error_band.json
  grid      적정가 표 → data/public/fair_price.parquet (개별 거래 없음, 모델 출력만)

실행: uv run python scripts/build_fair_price.py train | register | band | grid
"""

from __future__ import annotations

import json
import sys
import time

import duckdb
import numpy as np
import pandas as pd

from realestate.config import ROOT
from realestate.models import serve
from realestate.models.periods import FEATURE_DIR, PERIODS

OUT = ROOT / "data" / "public" / "fair_price.parquet"
GRID_MONTHS = 36  # 면적 타입: 정보 기준일까지 최근 36개월 정상 거래에 나온 면적


def train() -> None:
    t0 = time.perf_counter()
    meta = serve.train_and_save()
    meta.pop("cats")
    meta["sec_wall"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(meta, ensure_ascii=False))


def register() -> None:
    """MLflow: 실험 price_serve 에 학습 기록 + 모델 파일, Registry 에 apt_price_lgbm 새 버전."""
    import mlflow
    import mlflow.lightgbm

    from realestate.models import tracking

    txt, meta_path = serve.model_paths()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    serve.ascii_tempdir()
    tracking.use_experiment("price_serve")
    with mlflow.start_run(run_name=f"serve_D={meta['anchor']}") as run:
        mlflow.set_tags({**tracking.common_tags("serve"), "kind": "lgbm_serve", "params_hash": meta["params_hash"],
                         "asof": meta["anchor"], "target_ym": serve.TARGET_YM})  # fmt: skip
        cfg = json.loads(serve.LGBM_CONFIG.read_text(encoding="utf-8"))
        mlflow.log_params(cfg["params"])
        mlflow.log_metrics({k: float(meta[k]) for k in ("n_train", "best_iter", "valid_l1", "sec_total")})
        info = mlflow.lightgbm.log_model(
            serve.ServeModel.load(source="file").booster, name="model", registered_model_name=serve.MODEL_NAME
        )
        mlflow.log_artifact(str(meta_path), artifact_path="serve_files")  # 범주 목록 (예측에 필요)
        if serve.ERROR_BAND_PATH.exists():
            mlflow.log_artifact(str(serve.ERROR_BAND_PATH), artifact_path="serve_files")
        mv = info.registered_model_version
    meta["mlflow_run_id"], meta["registry_version"] = run.info.run_id, str(mv)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"MLflow run {run.info.run_id}, Registry {serve.MODEL_NAME} v{mv}")


def band() -> None:
    """평가 기간 실제 오차 ln(실거래 / 예측) 분위 (최근 12개월 단지·면적대 거래 수 구간별)."""
    cfg = json.loads(serve.LGBM_CONFIG.read_text(encoding="utf-8"))
    a, b = PERIODS["eval"]
    pred = (ROOT / "data" / "models" / "lgbm" / cfg["params_hash"] / "D=*[0-9].parquet").as_posix()
    feat = (FEATURE_DIR / "ym=*.parquet").as_posix()
    df = duckdb.sql(
        f"""
        SELECT f.cx12_n, ln(f.deal_amount / p.pred_lgbm) AS r
        FROM read_parquet('{feat}', union_by_name = true) f
        JOIN read_parquet('{pred}') p USING (trade_key)
        WHERE NOT f.is_canceled AND f.ym BETWEEN '{a}' AND '{b}'
        """
    ).df()
    df["grp"] = df["cx12_n"].map(serve.history_group)
    out = {"period": f"{a}~{b}", "params_hash": cfg["params_hash"], "residual": "ln(실거래 / 모델 예측)",
           "groups": {}}  # fmt: skip
    for name in [g for *_, g in serve.HISTORY_GROUPS] + ["전체"]:
        r = df["r"] if name == "전체" else df.loc[df["grp"] == name, "r"]
        q = r.quantile([0.05, 0.1, 0.5, 0.9, 0.95])
        out["groups"][name] = {
            "n": int(len(r)),
            "q05": round(float(q[0.05]), 5), "q10": round(float(q[0.1]), 5), "q50": round(float(q[0.5]), 5),
            "q90": round(float(q[0.9]), 5), "q95": round(float(q[0.95]), 5),
            "mdape_pct": round(float((np.exp(r) - 1).abs().median() * 100), 2),
        }  # fmt: skip
    serve.ERROR_BAND_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for k, v in out["groups"].items():
        lo, hi = (np.exp(v["q10"]) - 1) * 100, (np.exp(v["q90"]) - 1) * 100
        print(f"{k}: {v['n']:,}건, 오차 중위 {v['mdape_pct']}%, 80% 범위 {lo:+.1f}% ~ {hi:+.1f}%")


def grid_requests(con) -> pd.DataFrame:
    """적정가 표의 칸: 정보 기준일 C(T) 까지 공개된 최근 36개월 정상 거래의 (단지, 정수 ㎡ 면적) × 층 구간.

    면적은 그 정수 ㎡ 안에서 가장 많이 나온 실제 전용면적 (같으면 작은 값), 층 구간은 그 단지 최고층이 닿는 구간만
    (대표 층은 최고층 이하). 단지 정보(법정동·건축년도·최고층)는 API 와 같은 serve.complex_index.
    """
    from realestate.features import asof

    cx = serve.complex_index(con)
    c = asof.cutoff_date(serve.TARGET_YM).isoformat()
    keys = con.execute(
        f"""
        SELECT complex_key, area_int, area, n36 FROM (
            SELECT complex_key, round(area)::INT AS area_int, area,
                   sum(count(*)) OVER (PARTITION BY complex_key, round(area)::INT) AS n36,
                   row_number() OVER (PARTITION BY complex_key, round(area)::INT ORDER BY count(*) DESC, area) AS rn
            FROM h_serve WHERE deal_date > DATE '{c}' - INTERVAL {GRID_MONTHS} MONTH
            GROUP BY complex_key, area
        ) WHERE rn = 1
        """
    ).df()
    keys["area"] = (
        keys["area"].astype("float64").round(2)
    )  # 표에 적는 면적 그대로 예측 (API 에 같은 값을 넣으면 같은 결과)
    keys = keys.merge(cx, on="complex_key", how="left", validate="m:1")
    rows = []
    for name, lo, _hi, rep in serve.FLOOR_BUCKETS:
        ok = keys["max_floor"].fillna(1).clip(lower=1) >= lo
        k = keys[ok].copy()
        k["floor_band"] = name
        k["floor"] = np.maximum(lo, np.minimum(rep, k["max_floor"].fillna(rep))).astype(int)
        rows.append(k)
    req = pd.concat(rows, ignore_index=True)
    req["req_id"] = np.arange(len(req))
    return req


def grid() -> None:
    t0 = time.perf_counter()
    model = serve.ServeModel.load()
    bands = serve.load_error_band()["groups"]
    con = serve.open_history()
    req = grid_requests(con)
    print(f"칸 {len(req):,}개 (단지 {req['complex_key'].nunique():,}, 단지×면적 {req.groupby(['complex_key', 'area_int']).ngroups:,}) "
          f"{time.perf_counter() - t0:.0f}초")  # fmt: skip
    feats = serve.synth_features(con, req)
    con.close()
    fair = model.predict(feats)
    print(f"피처·예측 {time.perf_counter() - t0:.0f}초")
    grp = feats["cx12_n"].map(serve.history_group)
    q10 = grp.map(lambda g: bands[g]["q10"])
    q90 = grp.map(lambda g: bands[g]["q90"])
    out = req[
        ["complex_id", "apt_nm", "umd_nm", "sgg_cd", "build_year", "area", "floor_band", "floor"]
    ].assign(
        fair_price=np.round(fair).astype("int64"),
        fair_ppm2=np.round(fair / req["area"].to_numpy(float), 1),
        hist_group=grp.to_numpy(),
        range_low=np.round(fair * np.exp(q10.to_numpy(float))).astype("int64"),
        range_high=np.round(fair * np.exp(q90.to_numpy(float))).astype("int64"),
    )
    assert out["apt_nm"].notna().all(), "단지명이 없는 단지"
    ym = pd.Timestamp(f"{serve.TARGET_YM[:4]}-{serve.TARGET_YM[4:]}-01").date()
    out = (
        out.assign(area_m2=out["area"].astype("float64").round(2), floor_rep=out["floor"].astype("int16"),
                   build_year=out["build_year"].astype("Int16"), target_ym=ym, model_anchor=serve.ANCHOR)
        .drop(columns=["area", "floor"])
        .sort_values(["sgg_cd", "umd_nm", "apt_nm", "complex_id", "area_m2", "floor_rep"])
        .reset_index(drop=True)
    )[["complex_id", "sgg_cd", "umd_nm", "apt_nm", "build_year", "area_m2", "floor_band", "floor_rep",
       "fair_price", "fair_ppm2", "hist_group", "range_low", "range_high", "target_ym", "model_anchor"]]  # fmt: skip
    assert not out.duplicated(["complex_id", "area_m2", "floor_band"]).any()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"fair_price: {len(out):,}행 × {out.shape[1]}열, {OUT.stat().st_size / 2**20:.1f}MB, "
          f"{time.perf_counter() - t0:.0f}초")  # fmt: skip
    print("  " + ", ".join(f"{c}({t})" for c, t in out.dtypes.astype(str).items()))
    print(out["hist_group"].value_counts().to_string())
    print(out["fair_ppm2"].describe().round(1).to_string())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1]
    {"train": train, "band": band, "grid": grid, "register": register}[cmd]()


if __name__ == "__main__":
    main()
