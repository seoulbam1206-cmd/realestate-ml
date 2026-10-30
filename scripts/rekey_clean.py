"""정리본 trade_key·complex_key 다시 계산 (키 규칙을 바꿨을 때 한 번 실행).

2026-10-30: 동(apt_dong)을 키에서 빼고 단지명의 EUC-KR 밖 글자를 '?' 로 맞춤 (PLAN.md 6-1-2).
- CSV 파일: 키만 다시 계산
- API 파일: 이번에 받은 행(missing_count=0)과 이전에서 넘어온 행(missing_count≥1)을 따로 다시 계산.
  넘어온 행이 새 키로는 받은 행과 같은 거래면 빼고, 처음 본 시각은 더 이른 쪽을 남김
- apt_trade_change 의 옛 키 기록은 새 키와 맞지 않으므로 지움 (--keep-changes 로 남길 수 있음)
- 바뀐 키 때문에 생긴 중복이 없는지 파일마다 확인

실행: uv run python scripts/rekey_clean.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from realestate.config import APT_TRADE_CLEAN
from realestate.ingest import log as ingest_log
from realestate.ingest.molit_csv import add_keys


def rekey_api(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "missing_count" not in df:
        return add_keys(df.reset_index(drop=True)), 0
    fresh = add_keys(df[df["missing_count"] == 0].reset_index(drop=True))
    carry = df[df["missing_count"] > 0].reset_index(drop=True)
    if carry.empty:
        return fresh, 0
    carry = add_keys(carry)
    same = carry[carry["trade_key"].isin(fresh["trade_key"])]
    earlier = same.set_index("trade_key")["first_seen_at"]
    hit = fresh["trade_key"].isin(earlier.index)
    if hit.any():  # 빈 Series 에 map 하면 pandas 가 dtype 오류를 냄
        cur = fresh.loc[hit, "first_seen_at"]
        prev = pd.to_datetime(fresh.loc[hit, "trade_key"].map(earlier)).astype(cur.dtype)
        fresh.loc[hit, "first_seen_at"] = cur.where(cur <= prev, prev)
    out = pd.concat([fresh, carry[~carry["trade_key"].isin(fresh["trade_key"])]], ignore_index=True)
    return out, len(same)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-changes", action="store_true")
    args = parser.parse_args()

    n_files = n_rows = n_changed = n_merged = 0
    for path in sorted(APT_TRADE_CLEAN.glob("ym=*/source=*.parquet")):
        df = pd.read_parquet(path)
        old = set(df["trade_key"])
        if path.name == "source=csv.parquet":
            new, merged = add_keys(df.reset_index(drop=True)), 0
        else:
            new, merged = rekey_api(df)
        if new["trade_key"].duplicated().any():
            raise SystemExit(f"{path}: 새 키에 중복이 생김 — 중단 (아무것도 안 바뀐 파일만 저장됨)")
        n_files += 1
        n_rows += len(new)
        n_changed += len(set(new["trade_key"]) - old)
        n_merged += merged
        if not args.dry_run:
            tmp = path.with_suffix(".tmp")
            new.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(path)
    print(
        f"파일 {n_files}개, 행 {n_rows:,}, 키가 바뀐 행 {n_changed:,}, 같은 거래로 합친 넘어온 행 {n_merged:,}"
    )
    if not args.dry_run and not args.keep_changes:
        with ingest_log.connect() as con:
            n = con.execute("SELECT count(*) FROM apt_trade_change").fetchone()[0]
            con.execute("DELETE FROM apt_trade_change")
        print(f"apt_trade_change 옛 키 기록 {n:,}건 지움")
    if args.dry_run:
        print("(dry-run: 저장하지 않음)")


if __name__ == "__main__":
    main()
