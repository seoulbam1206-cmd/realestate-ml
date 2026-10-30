import numpy as np
import pandas as pd

from realestate.models.metrics import block_bootstrap_diff, metrics_by, price_metrics


def test_price_metrics_values():
    m = price_metrics(pd.Series([100, 200, 400]), pd.Series([110, 180, 400]))
    assert m["n"] == 3 and m["mae"] == 10
    assert np.isclose(m["mape"], (10 + 10 + 0) / 3) and m["mdape"] == 10 and np.isclose(m["within10"], 100)


def test_metrics_by_group():
    df = pd.DataFrame({"g": ["a", "a", "b"], "deal_amount": [100, 100, 200], "p": [90, 110, 100]})
    out = metrics_by(df, "g", {"P": "p"}).set_index("group")
    assert out.loc["a", "mae"] == 10 and out.loc["b", "mape"] == 50


def test_block_bootstrap_diff_sign_and_interval():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({"ym": rng.integers(0, 20, n), "deal_amount": 100.0})
    df["good"] = 100 + rng.normal(0, 5, n)
    df["bad"] = 100 + rng.normal(0, 20, n)
    out = block_bootstrap_diff(df, "bad", "good")  # good − bad < 0
    mae, lo, hi = out["mae"]
    assert mae < 0 and lo <= mae <= hi and hi < 0
