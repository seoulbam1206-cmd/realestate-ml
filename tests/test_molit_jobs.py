import os
import time
from datetime import date, datetime

import pandas as pd

from realestate.ingest import trades
from realestate.ingest.molit_jobs import plan_daily, shift_ym


def test_shift_ym():
    assert shift_ym("202610", -1) == "202609" and shift_ym("202601", -1) == "202512"
    assert shift_ym("202610", -12) == "202510"


def test_plan_daily_order_and_skips():
    today = date(2026, 10, 30)
    lawds = ["11110", "11140"]
    last_ok = {
        "sgg=11110,ym=202610": datetime(2026, 10, 30, 7),  # 오늘 이미 받음 → 건너뜀
        "sgg=11140,ym=202610": datetime(2026, 10, 29, 7),
        "sgg=11110,ym=202608": datetime(2026, 10, 20),  # 10일 전 → 아직 점검 안 함
        "sgg=11140,ym=202608": datetime(2026, 10, 16),  # 14일 전 → 점검
        "sgg=11110,ym=202607": datetime(2026, 10, 1),  # 가장 오래됨 → 안 받은 것 다음 맨 앞
    }
    jobs = plan_daily(today, lawds, last_ok)
    daily = [(j.ym, j.sgg) for j in jobs if j.job_type == "daily"]
    assert daily == [("202610", "11140"), ("202609", "11110"), ("202609", "11140")]
    rotate = [(j.ym, j.sgg) for j in jobs if j.job_type == "rotate"]
    assert len(rotate) == 11 * 2 - 1  # 2~12개월 전 × 2 시군구, 202608/11110 만 빠짐
    assert ("202608", "11110") not in rotate and ("202510", "11110") in rotate
    assert ("202509", "11110") not in rotate  # 13개월 전은 점검 대상 아님
    # 안 받은 것(최근 달부터) → 받은 지 오래된 것 순
    assert rotate[:2] == [("202607", "11140"), ("202606", "11110")] and rotate[-2:] == [
        ("202607", "11110"),
        ("202608", "11140"),
    ]
    assert jobs[: len(daily)] == [j for j in jobs if j.job_type == "daily"]


def _touch(path, t):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    os.utime(path, (t, t))


def test_month_sources_never_mix(tmp_path):
    now = time.time()
    lawds = ["11110", "11140"]
    _touch(tmp_path / "ym=202608" / "source=csv.parquet", now - 100)
    _touch(tmp_path / "ym=202608" / "source=api_sgg=11110.parquet", now)  # 일부만 → CSV
    _touch(tmp_path / "ym=202607" / "source=csv.parquet", now - 100)
    for sgg in lawds:  # 전부 + CSV 보다 새것 → API
        _touch(tmp_path / "ym=202607" / f"source=api_sgg={sgg}.parquet", now)
    _touch(tmp_path / "ym=202606" / "source=csv.parquet", now)  # CSV 를 나중에 다시 넣음 → CSV
    for sgg in lawds:
        _touch(tmp_path / "ym=202606" / f"source=api_sgg={sgg}.parquet", now - 100)
    _touch(tmp_path / "ym=202609" / "source=api_sgg=11110.parquet", now)  # CSV 없음, 일부만
    src = trades.month_sources(lawds, tmp_path).set_index("ym")
    assert src["use"].to_dict() == {
        "202606": "csv",
        "202607": "api",
        "202608": "csv",
        "202609": "api_partial",
    }
    files = trades.trade_files(lawds, ["202607", "202608"], tmp_path)
    assert sorted(p.name for p in files) == [
        "source=api_sgg=11110.parquet", "source=api_sgg=11140.parquet", "source=csv.parquet"
    ]  # fmt: skip
    assert isinstance(src.loc["202607", "files"], list) and pd.notna(src.loc["202609", "n_api"])
