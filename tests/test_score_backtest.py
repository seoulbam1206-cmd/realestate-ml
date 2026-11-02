"""매매 점수 검증 지표 (PLAN.md MVP 3-11)."""

import numpy as np
import pandas as pd
import pytest

from realestate.score import backtest as bt
from realestate.score.region import CONSIDER, NEVER, NODATA, NOT, REC


def _panel(months: int = 12, n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(months):
        t = f"2024{m + 1:02d}" if m < 12 else f"2025{m - 11:02d}"
        s = rng.normal(size=n)
        grade = np.where(s > 0.8, REC, np.where(s > 0, CONSIDER, NOT))
        grade[0] = NEVER
        for i in range(n):
            rows.append({"t": t, "sgg_cd": f"{i:05d}", "score": s[i], "P_A": -s[i], "grade": grade[i],
                         "X6": s[i] * 0.01, "R6": 0.0, "s_ret6": 0.0})  # fmt: skip
    return pd.DataFrame(rows)


def test_monthly_ic_perfect_and_reversed():
    d = _panel()
    assert bt.monthly_ic(d, "score", "X6").tolist() == pytest.approx([1.0] * 12)
    assert bt.monthly_ic(d, "P_A", "X6").tolist() == pytest.approx([-1.0] * 12)
    assert bt.monthly_ic(d.head(19), "score", "X6").empty  # 20개 미만 달은 빠짐


def test_matched_grades_keep_counts_and_never():
    d = _panel()
    g = bt.matched_grades(d, "P_A")
    for _t, part in d.assign(g=g).groupby("t"):
        assert (part["grade"].value_counts().sort_index() == part["g"].value_counts().sort_index()).all()
        assert (part.loc[part["grade"] == NEVER, "g"] == NEVER).all()
    assert bt.spread(d.assign(g=g), "X6", "g") < 0 < bt.spread(d, "X6")


def test_block_months_are_circular_blocks():
    months = [f"m{i}" for i in range(10)]
    for idx in bt.block_months(months, 3, 50, seed=1):
        assert len(idx) == 10
        for start in range(0, 9, 3):
            blk = idx[start : start + 3]
            assert all((b - a) % 10 == 1 for a, b in zip(blk, blk[1:], strict=False))


def test_bootstrap_on_perfect_signal():
    b = bt.bootstrap(_panel(), 6, "P_A", n=200)
    assert b["ic_mean"][0] == pytest.approx(1.0)
    assert b["ic_mean"][1] == pytest.approx(1.0)
    assert b["spread"][1] > 0
    assert b["ic_minus_P_A"][0] == pytest.approx(2.0)


def test_random_spreads_center_near_zero():
    r = bt.random_spreads(_panel(), "X6", n=200)
    assert abs(np.median(r)) < 0.003 < bt.spread(_panel(), "X6")


def test_nodata_kept_out_of_baselines_and_shown_in_table():
    d = _panel()
    d.loc[d["sgg_cd"] == "00001", "grade"] = NODATA
    g = bt.matched_grades(d, "P_A")
    assert (g[d["grade"] == NODATA] == NODATA).all()  # 다른 점수로 잘라도 데이터 부족은 그대로
    assert not (g[d["grade"] != NODATA] == NODATA).any()
    tbl = bt.grade_table(d, 6)
    assert tbl.loc[NODATA, "n"] == 12 and tbl.loc[NEVER, "n"] == 12
    d.loc[d["grade"] == NODATA, "X6"] = 100.0  # 섞기에 들어가면 차이가 크게 흔들림
    assert np.abs(bt.random_spreads(d, "X6", n=100)).max() < 1
