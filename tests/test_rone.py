import math
from pathlib import Path

import pandas as pd
import pytest

from realestate.config import DATA_RAW, sources
from realestate.ingest import rone

SURVEY_SAMPLE = """No,지역,지역,지역,지역,2024년 1월,2024년 1월,2024년 2월,2024년 2월,2024년 3월,2024년 3월
No,지역,지역,지역,지역,지수,지수,지수,지수,지수,지수
No,지역,지역,지역,지역,원자료,전기대비증감률,원자료,전기대비증감률,원자료,전기대비증감률
1,전국,전국,전국,전국,100.00,0.00,101.00,1.00,99.99,-1.00
2,서울,강북지역,도심권,종로구,,,50.00,0.00,51.00,2.00
3,경기,경부1권,안양시,만안구,80.00,0.00,80.00,0.00,80.00,0.00
4,경기,경부1권,안양시,안양시,70.00,0.00,70.00,0.00,70.00,0.00
"""
REAL_SAMPLE = """No,지역,지역,2024년 1월,2024년 2월
No,지역,지역,지수,지수
1,전국,전국,90.0,99.0
2,지방,지방,50.0,55.0
"""


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_bytes(body.encode("cp949"))
    return p


def test_read_wide_csv_detects_layout(tmp_path):
    w = rone.read_wide_csv(_write(tmp_path, "s.csv", SURVEY_SAMPLE))
    assert (w.header_rows, w.region_cols, w.values_per_month) == (3, 4, 2)
    assert w.months == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]

    r = rone.read_wide_csv(_write(tmp_path, "r.csv", REAL_SAMPLE))
    assert (r.header_rows, r.region_cols, r.values_per_month) == (2, 2, 1)


def test_region_key_removes_repeats_and_aliases():
    assert rone.region_key(["서울", "서울", "서울", "서울"]) == "서울"
    assert rone.region_key(["서울", "강북지역", "도심권", "도심권"]) == "서울>강북지역>도심권"
    assert rone.region_key(["지방", "지방"]) == "지방권"


def test_to_long_skips_blanks_and_blanks_first_mom(tmp_path):
    df = rone.to_long(rone.read_wide_csv(_write(tmp_path, "s.csv", SURVEY_SAMPLE)), "survey")
    jongno = df[df["region_key"] == "서울>강북지역>도심권>종로구"]
    assert jongno["ym"].min() == pd.Timestamp("2024-02-01")  # 빈 칸(통계 시작 전)은 행이 없다
    assert math.isnan(jongno["mom_pct"].iloc[0])  # 시작 달 증감률 0.00 은 '알 수 없음'
    assert jongno["mom_pct"].iloc[1] == 2.0


def test_real_index_mom_is_computed(tmp_path):
    df = rone.to_long(rone.read_wide_csv(_write(tmp_path, "r.csv", REAL_SAMPLE)), "real")
    nat = df[df["region_key"] == "전국"]
    assert nat["mom_pct"].iloc[1] == 10.0


def test_classify_types():
    keys = [
        "전국",
        "서울",
        "서울>강북지역",
        "서울>강북지역>도심권",
        "서울>강북지역>도심권>종로구",
        "경기",
        "경기>경부1권",
        "경기>경부1권>안양시",
        "경기>경부1권>안양시>만안구",
        "전남광주",
        "전남광주>광주",
        "전남광주>광주>동구",
    ]
    t = rone.classify(keys).set_index("region_key")
    assert t.loc["전국", "region_type"] == "묶음"
    assert t.loc["서울>강북지역>도심권", "region_type"] == "권역"
    assert t.loc["서울>강북지역>도심권>종로구", "region_type"] == "시군구"
    assert t.loc["경기>경부1권>안양시>만안구", "region_type"] == "일반구"
    assert t.loc["경기>경부1권>안양시>만안구", "display_name"] == "경기 안양시 만안구"
    assert t.loc["전남광주>광주", "region_type"] == "시도"
    assert t.loc["전남광주>광주", "display_name"] == "광주"
    assert t.loc["전남광주>광주>동구", "display_name"] == "광주 동구"


def test_duplicate_region_key_is_an_error(tmp_path):
    body = SURVEY_SAMPLE + "5,전국,전국,전국,전국,1.00,0.00,1.00,0.00,1.00,0.00\n"
    with pytest.raises(ValueError, match="겹칩니다"):
        rone.to_long(rone.read_wide_csv(_write(tmp_path, "s.csv", body)), "survey")


# ── 실제 파일 (data/raw 에 있을 때만) ──────────────────────────
_cfg = sources()["rone"]
_survey, _real = DATA_RAW / _cfg["survey_csv"], DATA_RAW / _cfg["real_csv"]
real_files = pytest.mark.skipif(not (_survey.exists() and _real.exists()), reason="R-ONE 원본 CSV 없음")


@pytest.fixture(scope="module")
def built():
    return rone.build(_survey, _real)


@real_files
def test_real_files_shape(built):
    index, region = built
    survey, real = index[index["source"] == "survey"], index[index["source"] == "real"]
    assert survey["region_key"].nunique() == 236
    assert real["region_key"].nunique() == 29
    assert set(real["region_key"]) <= set(survey["region_key"])  # 실거래 지역이 모두 조사 지역에 맞춰짐
    assert survey["ym"].min() == pd.Timestamp("2003-11-01")
    assert region["display_name"].is_unique
    assert region["region_key"].is_unique
    # 두 지수 모두 2026년 6월 = 100
    base = index[index["ym"] == pd.Timestamp("2026-06-01")]["index_value"]
    assert (base.round(1) == 100.0).all()


@real_files
def test_real_files_no_internal_gaps(built):
    index, _ = built
    for (_, _), g in index.groupby(["source", "region_key"]):
        months = pd.date_range(g["ym"].min(), g["ym"].max(), freq="MS")
        assert len(months) == len(g)
