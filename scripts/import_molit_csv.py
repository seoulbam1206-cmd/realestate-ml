"""data/raw/molit_csv/*.csv → data/clean/molit/apt_trade/ym=YYYYMM/source=csv.parquet

- 파일 안 '계약일자' 기간으로 월을 정한다 (파일 이름 월과 다르면 경고만)
- 같은 달 파일이 여러 개면 가장 늦게 받은 것만 쓴다
- 이미 같은 내용(sha256)으로 성공한 파일은 건너뛴다 (--force 로 다시)
- 결과는 모두 ingest_log 에 남는다

실행: uv run python scripts/import_molit_csv.py [--force]
"""

from __future__ import annotations

import argparse
from datetime import datetime

from realestate.config import APT_TRADE_CLEAN, BJD_RAW, MOLIT_CSV_RAW, REFERENCE_CLEAN
from realestate.ingest import bjd, molit_csv
from realestate.ingest import log as ingest_log

SOURCE = "molit_csv"


def load_bjd():
    codes = bjd.load(BJD_RAW)
    REFERENCE_CLEAN.mkdir(parents=True, exist_ok=True)
    codes.to_parquet(REFERENCE_CLEAN / "bjd_code.parquet", index=False)
    return bjd.name_lookup(codes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="이미 받은 파일도 다시 처리")
    args = parser.parse_args()

    files = sorted(MOLIT_CSV_RAW.glob("*.csv"))
    if not files:
        print(f"CSV 파일이 없습니다: {MOLIT_CSV_RAW}")
        return
    name_to_bjd = load_bjd()
    run_id = ingest_log.new_run_id()
    metas = [molit_csv.read_meta(p) for p in files]
    use, superseded = molit_csv.pick_latest_per_month(metas)

    unmatched_names: dict[str, int] = {}
    with ingest_log.connect() as con:
        done = set() if args.force else ingest_log.imported_hashes(con, SOURCE)
        for m in superseded:
            ingest_log.write(
                con, run_id=run_id, source=SOURCE, job_type="csv_import", target_key=f"ym={m.ym}",
                status="skipped", started_at=datetime.now(), source_file=m.path.name,
                period_from=m.period_from, period_to=m.period_to,
                warning="같은 달의 더 최근 다운로드 파일이 있어 건너뜀",
            )  # fmt: skip

        for m in use:
            started = datetime.now()
            sha = molit_csv.file_sha256(m.path)
            warning = (
                f"파일 이름 월({m.name_month})과 파일 안 계약일자 월({m.ym})이 다름 → 파일 안 기간 사용"
                if m.name_mismatch
                else None
            )
            common = dict(
                run_id=run_id, source=SOURCE, job_type="csv_import", target_key=f"ym={m.ym}",
                started_at=started, source_file=m.path.name, file_sha256=sha,
                period_from=m.period_from, period_to=m.period_to,
            )  # fmt: skip
            if sha in done:
                ingest_log.write(con, status="skipped", warning="이미 같은 파일을 처리함", **common)
                print(f"건너뜀  {m.ym}  (이미 처리)")
                continue
            try:
                molit_csv.validate_meta(m)
                df = molit_csv.parse(m, name_to_bjd)
                molit_csv.check_rows(m, df)
            except Exception as e:  # noqa: BLE001  실패도 로그에 남기고 다음 파일로
                ingest_log.write(con, status="fail", error_msg=str(e)[:500], warning=warning, **common)
                print(f"실패    {m.ym}  {e}")
                continue

            out_dir = APT_TRADE_CLEAN / f"ym={m.ym}"
            out_dir.mkdir(parents=True, exist_ok=True)
            df.drop(columns=["ym"]).to_parquet(
                out_dir / "source=csv.parquet", index=False, compression="zstd"
            )

            unmatched = df["bjd_cd"].isna()
            for name, n in df.loc[unmatched, "sgg_full_nm"].value_counts().items():
                unmatched_names[name] = unmatched_names.get(name, 0) + int(n)
            ingest_log.write(
                con, status="success", rows_fetched=len(df), rows_new=len(df),
                rows_canceled=int(df["is_canceled"].sum()), rows_unmatched=int(unmatched.sum()),
                warning=warning, **common,
            )  # fmt: skip
            flag = "  ⚠ 파일 이름 월 다름" if warning else ""
            print(
                f"완료    {m.ym}  {len(df):>6,}행  해제 {df['is_canceled'].sum():>5,}  "
                f"코드 미매칭 {unmatched.sum():>4,}{flag}"
            )

    if superseded:
        print(f"\n같은 달 중복 파일 {len(superseded)}개는 건너뜀 (최근 다운로드 사용)")
    if unmatched_names:
        print(f"\n법정동 코드와 이름이 안 맞은 주소 {len(unmatched_names)}종 (상위 15):")
        for name, n in sorted(unmatched_names.items(), key=lambda x: -x[1])[:15]:
            print(f"  {n:>6,}건  {name}")


if __name__ == "__main__":
    main()
