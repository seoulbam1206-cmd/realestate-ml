"""MLflow 로컬 기록 (PLAN.md MVP 3-8). 기록은 sqlite `mlflow.db`, 파일은 `mlruns/` (둘 다 git 제외).

UI: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import hashlib
import subprocess

import mlflow

from realestate.config import ROOT
from realestate.models import lgbm
from realestate.models.periods import FEATURE_DIR

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = ROOT / "mlruns"


def use_experiment(name: str) -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        mlflow.create_experiment(name, artifact_location=(ARTIFACT_ROOT / name).as_uri())
    return mlflow.set_experiment(name).experiment_id  # 새로 만든 실험도 활성으로 (안 하면 Default 에 쌓임)


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    return out.stdout.strip() + ("+dirty" if dirty.strip() else "")


def feature_meta_hash() -> str:
    return hashlib.sha1((FEATURE_DIR / "_meta.parquet").read_bytes()).hexdigest()[:10]


def common_tags(period: str) -> dict[str, str]:
    return {
        "period": period,
        "git_commit": git_commit(),
        "feature_set": lgbm.feature_set_hash(),
        "feature_table": f"{FEATURE_DIR.relative_to(ROOT).as_posix()} meta={feature_meta_hash()}",
    }
