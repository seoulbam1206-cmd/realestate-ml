"""경로와 설정. 비밀값은 .env 에서만 읽고, 비밀이 아닌 값은 config/sources.yaml 에서 읽는다."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
DATA_FEATURES = ROOT / "data" / "features"
RONE_CLEAN = DATA_CLEAN / "rone"
MOLIT_CSV_RAW = DATA_RAW / "molit_csv"
MOLIT_API_RAW = DATA_RAW / "molit" / "apt_trade"
APT_TRADE_CLEAN = DATA_CLEAN / "molit" / "apt_trade"
REFERENCE_CLEAN = DATA_CLEAN / "reference"
BJD_RAW = DATA_RAW / "법정동코드 전체자료.txt"
INGEST_DB = DATA_CLEAN / "ingest.duckdb"


@lru_cache
def sources() -> dict:
    with open(ROOT / "config" / "sources.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> None:
    """로컬 스크립트용. .env 를 환경변수로 올린다 (값은 절대 출력하지 않는다)."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"환경변수 {name} 가 비어 있습니다. .env 를 확인하세요.")
    return value
