"""로컬 API (/health, /region, /predict) 와 호가 판정 규칙."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from realestate.models import serve
from realestate.score import explain

PUBLIC = Path(__file__).resolve().parents[1] / "data" / "public"
HAS_REGION = (PUBLIC / "region_score.parquet").exists()
HAS_MODEL = serve.model_paths()[0].exists() and (PUBLIC / "fair_price.parquet").exists()


@pytest.fixture(scope="module")
def client():
    from realestate.api.app import app

    return TestClient(app)


def test_public_complex_id_is_stable_and_not_internal_key():
    a = serve.public_complex_id("252791b611b495a4")
    assert a == serve.public_complex_id("252791b611b495a4")
    assert a.startswith("c_") and len(a) == 12 and "252791b611" not in a


def test_floor_bucket_and_history_group():
    assert [serve.floor_bucket(f) for f in (1, 3, 4, 12, 13, 20, 21, 60)] == [
        "1~3층", "1~3층", "4~7층", "8~12층", "13~20층", "13~20층", "21층 이상", "21층 이상"
    ]  # fmt: skip
    assert [serve.history_group(n) for n in (0, float("nan"), 1, 4, 5, 19, 20)] == [
        "0건", "0건", "1~4건", "1~4건", "5~19건", "5~19건", "20건 이상"
    ]  # fmt: skip


def test_judge_uses_80pct_range_and_flags_few_history():
    j = serve.judge(110, 100, 90, 108, "5~19건")
    assert j == {"ask_gap_pct": 10.0, "judgement": "모델 대비 높음", "judgement_note": None}
    assert serve.judge(95, 100, 90, 108, "5~19건")["judgement"] == "모델과 비슷"
    assert serve.judge(80, 100, 90, 108, "5~19건")["judgement"] == "모델 대비 낮음"
    assert serve.judge(100, 100, 78, 125, "0건")["judgement_note"] == "이력이 적어 오차가 큼"


def test_error_band_file_matches_groups():
    bands = serve.load_error_band()["groups"]
    assert set(bands) == {g for *_, g in serve.HISTORY_GROUPS} | {"전체"}
    for b in bands.values():
        assert b["q10"] < 0 < b["q90"] and b["n"] > 1000
    assert bands["0건"]["q90"] - bands["0건"]["q10"] > bands["20건 이상"]["q90"] - bands["20건 이상"]["q10"]


def test_notices_never_say_recommend():
    assert "추천" not in serve.ASK_NOTE
    assert "단지·개별 단위는 검증하지 않은 참고 정보입니다." in serve.ASK_NOTE


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["model"]["target_ym"] == "2026-10"


@pytest.mark.skipif(not HAS_REGION, reason="build_public.py 를 먼저 실행")
def test_region(client):
    r = client.get("/region/11680")
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "지역 등급(시군구 단위)"
    assert body["notice"] == explain.REGION_NOTE
    assert body["grade"] in ["매매 추천", "매매 고려", "매매 고려 대상 아님", "매매 절대 불가", "데이터 부족"]
    assert set(body["factors"]) == set(explain.FACTORS.values())
    assert client.get("/region/99999").status_code == 404
    assert client.get("/region/abc").status_code == 422


def test_predict_rejects_bad_input(client):
    ok = {"complex_id": "c_0123456789", "area_m2": 84.9, "floor": 10}
    assert client.post("/predict", json={**ok, "complex_id": "1234"}).status_code == 422
    assert client.post("/predict", json={**ok, "area_m2": 0}).status_code == 422
    assert client.post("/predict", json={**ok, "floor": 0}).status_code == 422
    assert client.post("/predict", json={**ok, "ask_price_manwon": -5}).status_code == 422


@pytest.mark.skipif(not HAS_MODEL, reason="build_fair_price.py train·grid 를 먼저 실행")
def test_predict_matches_fair_price_table(client):
    """표의 칸과 같은 면적·층이면 API 값 = 표 값 (같은 모델·같은 피처 SQL). 첫 호출은 이력 로딩으로 느림."""
    fair = pd.read_parquet(PUBLIC / "fair_price.parquet")
    picks = pd.concat(
        [fair[fair["hist_group"] == g].sample(2, random_state=0) for g in fair["hist_group"].unique()]
    )
    for r in picks.itertuples():
        body = {"complex_id": r.complex_id, "area_m2": r.area_m2, "floor": int(r.floor_rep),
                "ask_price_manwon": int(r.fair_price * 1.3)}  # fmt: skip
        out = client.post("/predict", json=body).json()
        assert abs(out["fair_price_manwon"] - r.fair_price) <= max(1, r.fair_price * 1e-4), (r, out)
        assert out["range_80"]["history_group"] == r.hist_group
        assert out["price_basis"] == "2026년 10월 기준 적정가" and out["model"]["target_ym"] == "2026-10"
        assert out["judgement"] == "모델 대비 높음"
        assert (out["judgement_note"] == "이력이 적어 오차가 큼") == (r.hist_group == "0건")
        assert out["region"]["label"] == "지역 등급(시군구 단위)"
        assert out["complex"]["apt_nm"] == r.apt_nm
    assert (
        client.post("/predict", json={"complex_id": "c_0000000000", "area_m2": 84.9, "floor": 3}).status_code
        == 422
    )
