"""LightGBM 시세 모델: 실측 → 튜닝(T 2023 만) → 설정 고정 → 평가 1회 (PLAN.md MVP 3-2, 3-5, 3-7, 3-8)

- tune: 후보 × 튜닝 폴드(T 2023.3 / 7 / 11). 폴드 결과는 reports/lgbm_tuning.csv 에 하나씩 남겨 끊겨도 이어서.
        끝나면 3폴드 평균 MAPE 가 가장 낮은 후보를 config/lgbm.json 에 고정 (MAE 는 함께 보고)
- backtest: 고정 설정으로 평가 T 2024.1~2025.9 (기준일 7개) + 확인 T 2025.10~2026.8 (4개, 참고).
        기준일 하나 = 프로세스 하나 (anchor 명령). 같은 설정·같은 기간의 평가가 MLflow 에 이미 있으면 거부.
        --periods eval 이면 평가 기준일만 실행·기록 (확인은 나중에 따로)
- prep: 같은 고정 설정으로 T 2021.10~2023.12 예측 (기준일 9개). 매매 점수 B 요소의 튜닝 기간 원값용 (MVP 3-11).
        시세 모델 성적에는 안 씀 (평가·리포트는 eval·check 기준일만 읽음)
- report: B0·B1·LightGBM 같은 표 → reports/lgbm.md

실행:
  uv run python scripts/train_lgbm.py probe      # 학습 1회 시간·최대 메모리 (튜닝 폴드 D = 2023.9.30, 기본 설정)
  uv run python scripts/train_lgbm.py tune
  uv run python scripts/train_lgbm.py backtest [--periods eval]
  uv run python scripts/train_lgbm.py report [--periods eval]
  uv run python scripts/train_lgbm.py prep
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import date, datetime

import mlflow
import pandas as pd
import psutil

from realestate.config import ROOT
from realestate.features import asof
from realestate.features.trade_features import FEATURES
from realestate.models import lgbm, tracking
from realestate.models.metrics import price_metrics
from realestate.models.periods import FEATURE_DIR, PERIODS

TUNING_FOLDS = ["202303", "202307", "202311"]  # T 월. 모델 기준일은 직전 분기 말일
TUNING_CSV = ROOT / "reports" / "lgbm_tuning.csv"
CONFIG = ROOT / "config" / "lgbm.json"
OUT = ROOT / "data" / "models" / "lgbm"
# 후보 (MVP 3-5: 20개 이하). 학습 창은 튜닝에서 가릴 수 없음: 학습이 2021.1 부터라 폴드 학습 기간이 23~33개월
# → 최근 36개월 창 = 확장 창. 확장 창으로 고정
CANDIDATES = {
    f"c{i:02d}": {"objective": obj, "num_leaves": nl, "min_data_in_leaf": md}
    for i, (obj, nl, md) in enumerate(
        (o, n, m) for o in ("l2", "huber", "l1") for n in (63, 255) for m in (20, 200)
    )
}
for _c in CANDIDATES.values():
    if _c["objective"] == "huber":
        _c["alpha"] = 0.1  # log 오차 0.1 ≈ 10% 를 넘는 잔차는 선형으로


def peak_mb() -> float:
    """이 프로세스의 최대 작업 집합 (Windows peak_wset, 없으면 현재 RSS)."""
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**20, 0)


def scoring_rows(first: str, last: str) -> pd.DataFrame:
    """채점 행: 지금 기준 해제되지 않은 거래 (기준선과 같은 행)."""
    df = lgbm.load_months(first, last, [*lgbm.LOAD_COLS, "trade_key", "b0_ppm2", "b1_ppm2"])
    return df[~df["is_canceled"]].reset_index(drop=True)


def probe() -> None:
    t_ym = TUNING_FOLDS[-1]
    d = lgbm.anchor_for(t_ym)
    t0 = time.perf_counter()
    train = lgbm.training_frame(d)
    t_load = time.perf_counter() - t0
    mem_load = peak_mb()
    model, cats, info = lgbm.fit(train, d, {})
    del train
    gc.collect()
    test = scoring_rows(t_ym, t_ym)
    t1 = time.perf_counter()
    pred = lgbm.predict_price(model, test, cats)
    m = price_metrics(test["deal_amount"], pred)
    out = {
        "anchor": str(d), "t": t_ym, **info, "sec_load": round(t_load, 1),
        "sec_predict": round(time.perf_counter() - t1, 1), "peak_mb_after_load": mem_load,
        "peak_mb": peak_mb(), "threads": lgbm.NUM_THREADS, "fold_mae": round(m["mae"]), "fold_mape": round(m["mape"], 2),
    }  # fmt: skip
    print(json.dumps(out, ensure_ascii=False))


def row_counts() -> None:
    """기준일별 학습 행 수 (시간 추정용. 가격·지표는 보지 않음)."""
    anchors = sorted({lgbm.anchor_for(f"{y}{m:02d}") for y in range(2023, 2027) for m in (1, 4, 7, 10)})
    allrows = lgbm.load_months(lgbm.TRAIN_FROM, "202609", ["ym", "deal_date", "cdeal_day"])
    for d in anchors:
        print(d, len(asof.training_rows(allrows, d)))


def check_features() -> None:
    asof.audit(pd.read_parquet(FEATURE_DIR / "_meta.parquet"), FEATURES)  # 시점 규칙 위반이면 멈춤


# ---------------------------------------------------------------- 튜닝 (T 2023 만)


def tune() -> None:
    check_features()
    done = pd.read_csv(TUNING_CSV, dtype={"t": str}) if TUNING_CSV.exists() else pd.DataFrame()
    for t_ym in TUNING_FOLDS:
        assert t_ym <= PERIODS["tuning"][1]
        todo = [c for c in CANDIDATES if done.empty or not ((done["cand"] == c) & (done["t"] == t_ym)).any()]
        if not todo:
            continue
        d = lgbm.anchor_for(t_ym)
        train, test = lgbm.training_frame(d), scoring_rows(t_ym, t_ym)
        assert train["deal_date"].max() <= pd.Timestamp(PERIODS["tuning"][1][:4] + "-12-31")
        for c in todo:
            model, cats, info = lgbm.fit(train, d, CANDIDATES[c])
            m = price_metrics(test["deal_amount"], lgbm.predict_price(model, test, cats))
            row = {"cand": c, "t": t_ym, "anchor": str(d), **info, **m, "peak_mb": peak_mb()}
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(TUNING_CSV, index=False)
            print(
                f"{c} T={t_ym}: MAE {m['mae']:,.0f} MAPE {m['mape']:.2f} 트리 {info['best_iter']} {info['sec_total']}초"
            )
            del model
        del train, test
        gc.collect()
    log_tuning(done)
    choose(done)


def tuning_summary(done: pd.DataFrame) -> pd.DataFrame:
    full = done.groupby("cand").filter(lambda g: set(g["t"]) == set(TUNING_FOLDS))
    return (
        full.groupby("cand")
        .agg(
            mae=("mae", "mean"),
            mape=("mape", "mean"),
            mdape=("mdape", "mean"),
            within10=("within10", "mean"),
            best_iter=("best_iter", "mean"),
            sec=("sec_total", "mean"),
        )  # fmt: skip
        .sort_values(["mape", "mae"])
    )


def log_tuning(done: pd.DataFrame) -> None:
    tracking.use_experiment("price_tuning")
    logged = mlflow.search_runs(filter_string="tags.kind = 'lgbm_candidate'", output_format="pandas")
    seen = set(logged["tags.params_hash"]) if len(logged) else set()
    tags = tracking.common_tags("tuning")
    for c, s in tuning_summary(done).iterrows():
        params = {**lgbm.BASE_PARAMS, **CANDIDATES[c]}
        if lgbm.params_hash(params) in seen:
            continue
        with mlflow.start_run(run_name=f"lgbm_{c}"):
            mlflow.set_tags({**tags, "kind": "lgbm_candidate", "cand": c, "params_hash": lgbm.params_hash(params),
                             "window": "expanding_from_202101"})  # fmt: skip
            mlflow.log_params(params)
            for r in done[done["cand"] == c].itertuples():
                for k in ("mae", "mape", "mdape", "within10", "best_iter", "sec_total"):
                    mlflow.log_metric(f"{k}_T{r.t}", getattr(r, k))
            mlflow.log_metrics({f"mean_{k}": float(s[k]) for k in ("mae", "mape", "mdape", "within10")})


def choose(done: pd.DataFrame) -> None:
    s = tuning_summary(done)
    if len(s) < len(CANDIDATES):
        print(f"후보 {len(s)}/{len(CANDIDATES)}개만 끝남 → 설정 고정 안 함")
        return
    best = s.index[0]
    params = {**lgbm.BASE_PARAMS, **CANDIDATES[best]}
    CONFIG.write_text(
        json.dumps(
            {"cand": best, "params": params, "params_hash": lgbm.params_hash(params), "window": "expanding_from_202101",
             "rule": "튜닝 3폴드(T 2023.3/7/11) 평균 MAPE 최소", "fixed_at": f"{datetime.now():%Y-%m-%d %H:%M}",
             "tuning_mean_mae": round(float(s.loc[best, "mae"])), "tuning_mean_mape": round(float(s.loc[best, "mape"]), 3)},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    print(s.round(3).to_string())
    print(f"고정: {best} → {CONFIG.relative_to(ROOT)}")


# ---------------------------------------------------------------- 평가 (설정 고정 뒤 1회)


BACKTEST_PERIODS = ("eval", "check")
PREP = ("202110", "202312")  # 점수 B 요소용 walk-forward 예측 (첫 모델 D 2021.9.30 은 학습 8개월)


def backtest_months(periods: tuple[str, ...] = BACKTEST_PERIODS) -> dict[date, tuple[str, list[str]]]:
    """기준일 → (기간, 그 모델로 예측할 T 월들)."""
    out: dict[date, tuple[str, list[str]]] = {}
    spans = {**PERIODS, "prep": PREP}
    for period in periods:
        a, b = spans[period]
        for p in pd.period_range(f"{a[:4]}-{a[4:]}", f"{b[:4]}-{b[4:]}", freq="M"):
            ym = f"{p.year}{p.month:02d}"
            out.setdefault(lgbm.anchor_for(ym), (period, []))[1].append(ym)
    return out


def run_anchor(d: date, months: list[str], cfg: dict) -> None:
    """기준일 하나: 학습 → T 월 예측을 파일로 (프로세스 하나에서, 끝나면 메모리 반환)."""
    t0 = time.perf_counter()
    train = lgbm.training_frame(d)
    assert train["deal_date"].max() <= pd.Timestamp(d)
    model, cats, info = lgbm.fit(train, d, cfg["params"])
    del train
    gc.collect()
    test = scoring_rows(months[0], months[-1])
    test = test[test["ym"].isin(months)]
    pred = pd.DataFrame(
        {"trade_key": test["trade_key"], "ym": test["ym"], "pred_lgbm": lgbm.predict_price(model, test, cats)}
    )
    out = OUT / cfg["params_hash"]
    out.mkdir(parents=True, exist_ok=True)
    stem = f"D={d:%Y%m%d}"
    pred.to_parquet(out / f"{stem}.parquet", index=False)
    imp = pd.DataFrame({"feature": model.feature_name(), "gain": model.feature_importance("gain")})
    imp.to_csv(out / f"{stem}_importance.csv", index=False)
    info |= {
        "anchor": str(d),
        "months": months,
        "sec_wall": round(time.perf_counter() - t0, 1),
        "peak_mb": peak_mb(),
    }
    (out / f"{stem}.json").write_text(json.dumps(info), encoding="utf-8")
    print(json.dumps(info))


def backtest(force: bool, periods: tuple[str, ...]) -> None:
    check_features()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    tracking.use_experiment("price_backtest")
    prev = mlflow.search_runs(
        filter_string=f"tags.kind = 'lgbm_backtest' and tags.params_hash = '{cfg['params_hash']}' "
        f"and tags.periods = '{','.join(periods)}'",
        output_format="pandas",
    )
    if len(prev) and not force:
        raise SystemExit(
            f"이 설정({cfg['params_hash']})의 평가가 이미 기록됨: run {prev['run_id'].iloc[0]} (다시 하려면 --force)"
        )
    run_anchors(cfg, periods)
    log_backtest(cfg, periods)


def run_anchors(cfg: dict, periods: tuple[str, ...]) -> None:
    """기준일마다 프로세스 하나. 결과 json 이 있는 기준일은 건너뜀 (끊겨도 다시 실행하면 이어짐)."""
    out = OUT / cfg["params_hash"]
    for d, (_, months) in backtest_months(periods).items():
        if (out / f"D={d:%Y%m%d}.json").exists():
            continue
        subprocess.run(
            [sys.executable, __file__, "anchor", "--d", str(d), "--months", ",".join(months)],
            check=True,
            cwd=ROOT,
        )


def prep() -> None:
    check_features()
    run_anchors(json.loads(CONFIG.read_text(encoding="utf-8")), ("prep",))


def load_backtest(
    cfg: dict, periods: tuple[str, ...] = BACKTEST_PERIODS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = OUT / cfg["params_hash"]
    stems = [f"D={d:%Y%m%d}" for d in backtest_months(periods)]
    runs = pd.DataFrame([json.loads((out / f"{s}.json").read_text(encoding="utf-8")) for s in stems])
    preds = pd.concat(pd.read_parquet(out / f"{s}.parquet") for s in stems)
    a, b = PERIODS[periods[0]][0], PERIODS[periods[-1]][1]
    cols = [
        "trade_key",
        "ym",
        "deal_amount",
        "area",
        "dealing_gbn",
        "is_canceled",
        "b0_ppm2",
        "b1_ppm2",
        "b1_level",
    ]
    df = lgbm.load_months(a, b, cols)
    df = df[~df["is_canceled"]].merge(preds, on=["trade_key", "ym"], how="left", validate="1:1")
    assert df["pred_lgbm"].notna().all(), "예측이 빠진 채점 행"
    df["pred_b0"], df["pred_b1"] = df["b0_ppm2"] * df["area"], df["b1_ppm2"] * df["area"]
    df["period"] = df["ym"].map(lambda ym: "eval" if ym <= PERIODS["eval"][1] else "check")
    return df, runs


def log_backtest(cfg: dict, periods: tuple[str, ...] = BACKTEST_PERIODS) -> None:
    from realestate.models.metrics import block_bootstrap_diff

    df, runs = load_backtest(cfg, periods)
    out = OUT / cfg["params_hash"]
    tracking.use_experiment("price_backtest")
    tags = tracking.common_tags("eval")
    with mlflow.start_run(run_name=f"lgbm_{cfg['cand']}"):
        mlflow.set_tags(
            {
                **tags,
                "kind": "lgbm_backtest",
                "params_hash": cfg["params_hash"],
                "window": cfg["window"],
                "periods": ",".join(periods),
            }  # fmt: skip
        )
        mlflow.log_params(cfg["params"])
        for r in runs.itertuples():
            part = df[df["ym"].isin(r.months)]
            period = "eval" if r.months[0] <= PERIODS["eval"][1] else "check"
            with mlflow.start_run(run_name=f"D={r.anchor}", nested=True):
                mlflow.set_tags({"period": period, "asof": r.anchor, "data_rows": str(r.n_train)})
                mlflow.log_metrics(
                    {
                        k: float(getattr(r, k))
                        for k in ("n_train", "best_iter", "sec_total", "sec_wall", "peak_mb")
                    }
                )
                mlflow.log_metrics(
                    {k: v for k, v in price_metrics(part["deal_amount"], part["pred_lgbm"]).items()}
                )
        for period in periods:
            p = df[df["period"] == period]
            mlflow.log_metrics(
                {f"{period}_{k}": v for k, v in price_metrics(p["deal_amount"], p["pred_lgbm"]).items()}
            )
        ev = df[df["period"] == "eval"]
        for base in ("b0", "b1"):
            boot = block_bootstrap_diff(ev, f"pred_{base}", "pred_lgbm")
            for k, (mid, lo, hi) in boot.items():
                mlflow.log_metrics(
                    {
                        f"eval_{k}_minus_{base}": mid,
                        f"eval_{k}_minus_{base}_lo": lo,
                        f"eval_{k}_minus_{base}_hi": hi,
                    }
                )
        for d in backtest_months(periods):
            for f in sorted(out.glob(f"D={d:%Y%m%d}*")):
                mlflow.log_artifact(str(f), artifact_path="anchors")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "rows", "tune", "backtest", "anchor", "report", "prep"])
    ap.add_argument("--d")
    ap.add_argument("--months")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--periods", default=",".join(BACKTEST_PERIODS), help="eval 또는 eval,check")
    args = ap.parse_args()
    periods = tuple(args.periods.split(","))
    assert periods in (("eval",), BACKTEST_PERIODS), periods
    if args.cmd == "anchor":
        run_anchor(
            date.fromisoformat(args.d), args.months.split(","), json.loads(CONFIG.read_text(encoding="utf-8"))
        )
    elif args.cmd == "backtest":
        backtest(args.force, periods)
    elif args.cmd == "report":
        from report_lgbm import write_report  # scripts/ 안의 리포트

        write_report(json.loads(CONFIG.read_text(encoding="utf-8")), periods)
    else:
        {"probe": probe, "rows": row_counts, "tune": tune, "prep": prep}[args.cmd]()


if __name__ == "__main__":
    main()
