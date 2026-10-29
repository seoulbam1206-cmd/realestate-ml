"""R-ONE CSV 2개 → data/clean/rone/*.parquet

실행: uv run python scripts/build_rone.py
"""

from __future__ import annotations

from realestate.config import DATA_RAW, RONE_CLEAN, sources
from realestate.ingest import rone


def main() -> None:
    cfg = sources()["rone"]
    index, region = rone.build(DATA_RAW / cfg["survey_csv"], DATA_RAW / cfg["real_csv"], cfg["encoding"])

    RONE_CLEAN.mkdir(parents=True, exist_ok=True)
    index.to_parquet(RONE_CLEAN / "rone_index.parquet", index=False, compression="zstd")
    region.to_parquet(RONE_CLEAN / "rone_region.parquet", index=False, compression="zstd")

    for src, g in index.groupby("source"):
        print(
            f"[{src}] 지역 {g['region_key'].nunique()}개, {len(g):,}행, {g['ym'].min():%Y-%m} ~ {g['ym'].max():%Y-%m}"
        )
    print("지역 유형:", region["region_type"].value_counts().to_dict())
    print(f"저장: {RONE_CLEAN}")


if __name__ == "__main__":
    main()
