"""data/clean 의 공개용 집계표를 TiDB 에 올린다 (쓰기 계정, .env 의 TIDB_*).

실행:
  uv run python scripts/publish.py --dry-run   # 접속 없이 올릴 행 수만 확인
  uv run python scripts/publish.py             # 실제 업로드
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import text

from realestate.config import RONE_CLEAN, load_env, require_env
from realestate.db import tidb

TABLES = {
    "rone_region": RONE_CLEAN / "rone_region.parquet",
    "rone_index": RONE_CLEAN / "rone_index.parquet",
}
CHUNK = 5000


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date
    rows = df.to_dict("records")
    for r in rows:  # NaN → NULL
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    frames = {name: pd.read_parquet(path) for name, path in TABLES.items()}
    for name, df in frames.items():
        print(f"{name}: {len(df):,}행")
    if args.dry_run:
        return

    load_env()
    engine = tidb.make_engine(
        require_env("TIDB_HOST"),
        int(require_env("TIDB_PORT")),
        require_env("TIDB_USER"),
        require_env("TIDB_PASSWORD"),
        require_env("TIDB_DATABASE"),
    )
    now = datetime.now(UTC).replace(tzinfo=None)
    with engine.begin() as conn:  # 전부 성공하거나 전부 취소
        for stmt in tidb.schema_sql():
            conn.execute(text(stmt))
        for name, df in frames.items():
            rows = _records(df)
            cols = list(df.columns)
            insert = text(
                f"INSERT INTO {name} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})"
            )
            conn.execute(text(f"DELETE FROM {name}"))
            for i in range(0, len(rows), CHUNK):
                conn.execute(insert, rows[i : i + CHUNK])
            last_ym = df["ym"].max().date() if "ym" in df else None
            conn.execute(
                text(
                    "REPLACE INTO publish_meta (table_name, row_count, data_last_ym, published_at) "
                    "VALUES (:t, :n, :ym, :at)"
                ),
                {"t": name, "n": len(df), "ym": last_ym, "at": now},
            )
            print(f"업로드 완료: {name} {len(df):,}행")


if __name__ == "__main__":
    main()
