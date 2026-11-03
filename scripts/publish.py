"""data/clean 의 공개용 집계표를 TiDB 에 올린다 (쓰기 계정, .env 의 TIDB_*).

실행:
  uv run python scripts/publish.py --dry-run                 # 접속 없이 올릴 표·열·행 수만 확인
  uv run python scripts/publish.py                           # 전부 업로드
  uv run python scripts/publish.py --tables mvp3 --dry-run   # MVP 3 표만 (build_public.py 결과)
  uv run python scripts/publish.py --tables complex --dry-run  # 단지 시세 표만 (build_complex_price.py 결과)
  uv run python scripts/publish.py --tables fair --dry-run     # 호가 비교 적정가 표 (build_fair_price.py grid)

TiDB 의 표 열이 올릴 표와 다르면 (schema.sql 을 바꾼 뒤) 그 표만 지우고 schema.sql 대로 다시 만든다.
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import text

from realestate.config import RONE_CLEAN, ROOT, load_env, require_env
from realestate.db import tidb

PUBLIC = ROOT / "data" / "public"
GROUPS = {
    "rone": {
        "rone_region": RONE_CLEAN / "rone_region.parquet",
        "rone_index": RONE_CLEAN / "rone_index.parquet",
    },
    "mvp3": {
        name: PUBLIC / f"{name}.parquet"
        for name in (
            "region_name",
            "agg_region_month",
            "region_score",
            "bt_ic",
            "bt_grade",
            "bt_baseline",
            "bt_meta",
        )
    },
    "complex": {"complex_price": PUBLIC / "complex_price.parquet"},  # scripts/build_complex_price.py
    "fair": {"fair_price": PUBLIC / "fair_price.parquet"},  # scripts/build_fair_price.py grid
}
TABLES = {k: v for g in GROUPS.values() for k, v in g.items()}
# 개별 거래 행을 실수로 올리지 않도록: 거래 단위 열이 있는 표는 거부
FORBIDDEN_COLS = {"trade_key", "deal_amount", "deal_date", "apt_seq", "complex_key", "floor", "area", "ppm2"}
CHUNK = 5000
BIG = 100_000  # 이보다 큰 표는 임시 표에 나눠 넣고 이름을 한 번에 바꿈 (TiDB 트랜잭션 크기 한도)


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


def publish_big(engine, name: str, df: pd.DataFrame, now: datetime) -> None:
    """임시 표 {name}__new 에 조각마다 확정하며 넣고, 행 수를 확인한 뒤 RENAME 으로 한 번에 바꿈.

    앱은 바꾸기 전까지 기존 표를, 바꾼 뒤에는 새 표를 읽음 (중간 상태를 보지 않음).
    """
    new, old = f"{name}__new", f"{name}__old"
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {new}"))
        conn.execute(text(f"DROP TABLE IF EXISTS {old}"))
        conn.execute(text(create_stmt(name).replace(f"EXISTS {name} ", f"EXISTS {new} ", 1)))
    rows, cols = _records(df), list(df.columns)
    insert = text(f"INSERT INTO {new} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})")
    for i in range(0, len(rows), CHUNK):
        with engine.begin() as conn:
            conn.execute(insert, rows[i : i + CHUNK])
        if (i // CHUNK) % 20 == 0:
            print(f"  {name}: {min(i + CHUNK, len(rows)):,}/{len(rows):,}")
    with engine.begin() as conn:
        n = conn.execute(text(f"SELECT count(*) FROM {new}")).scalar_one()
        assert n == len(df), f"{new}: {n}행 (기대 {len(df)})"
        conn.execute(text(f"RENAME TABLE {name} TO {old}, {new} TO {name}"))
        conn.execute(text(f"DROP TABLE {old}"))
    ym_col = next((c for c in ("ym", "asof_ym", "window_to", "target_ym") if c in df), None)
    with engine.begin() as conn:
        conn.execute(
            text(
                "REPLACE INTO publish_meta (table_name, row_count, data_last_ym, published_at) "
                "VALUES (:t, :n, :ym, :at)"
            ),
            {
                "t": name,
                "n": len(df),
                "ym": pd.Timestamp(df[ym_col].max()).date() if ym_col else None,
                "at": now,
            },
        )
    print(f"업로드 완료: {name} {len(df):,}행 (임시 표 → 이름 바꿈)")


def create_stmt(name: str) -> str:
    return next(s for s in tidb.schema_sql() if s.startswith(f"CREATE TABLE IF NOT EXISTS {name} "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tables", choices=["all", *GROUPS], default="all")
    args = parser.parse_args()

    chosen = TABLES if args.tables == "all" else GROUPS[args.tables]
    frames = {name: pd.read_parquet(path) for name, path in chosen.items()}
    for name, df in frames.items():
        bad = FORBIDDEN_COLS & set(df.columns)
        if bad:
            raise SystemExit(f"{name}: 거래 단위 열 {sorted(bad)} 이 있어 올리지 않음")
        print(f"{name}: {len(df):,}행 × {df.shape[1]}열 — {', '.join(df.columns)}")
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
        # 표 구조 맞추기 (DDL 은 TiDB 에서 바로 확정되므로 데이터 넣기 전에 끝냄)
        for name, df in frames.items():
            have = (
                conn.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = DATABASE() AND table_name = :t ORDER BY ordinal_position"
                    ),
                    {"t": name},
                )
                .scalars()
                .all()
            )
            if [c.lower() for c in have] != [c.lower() for c in df.columns]:
                print(f"{name}: 열이 바뀜 {list(have)} → {list(df.columns)} → 표를 다시 만듦")
                conn.execute(text(f"DROP TABLE {name}"))
                conn.execute(text(create_stmt(name)))
        for name, df in frames.items():
            if len(df) > BIG:
                continue  # 큰 표는 아래에서 따로 (트랜잭션 크기 한도)
            rows = _records(df)
            cols = list(df.columns)
            insert = text(
                f"INSERT INTO {name} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})"
            )
            conn.execute(text(f"DELETE FROM {name}"))
            for i in range(0, len(rows), CHUNK):
                conn.execute(insert, rows[i : i + CHUNK])
            ym_col = next((c for c in ("ym", "asof_ym", "window_to", "target_ym") if c in df), None)
            last_ym = pd.Timestamp(df[ym_col].max()).date() if ym_col else None
            conn.execute(
                text(
                    "REPLACE INTO publish_meta (table_name, row_count, data_last_ym, published_at) "
                    "VALUES (:t, :n, :ym, :at)"
                ),
                {"t": name, "n": len(df), "ym": last_ym, "at": now},
            )
            print(f"업로드 완료: {name} {len(df):,}행")
    for name, df in frames.items():
        if len(df) > BIG:
            publish_big(engine, name, df, now)


if __name__ == "__main__":
    main()
