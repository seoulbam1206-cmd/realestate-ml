"""화면 규칙 검사: 모든 페이지에 출처 푸터, 순위·점수 페이지에 면책 문구 (PLAN.md 10장)."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from realestate.config import RONE_CLEAN

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]
DISCLAIMER_PAGES = {"2_하락_지역_순위.py", "4_지역_점수_등급.py", "5_백테스트_성적표.py", "6_단지_시세.py"}
MVP3_PAGES = {
    "3_실거래_집계.py",
    "4_지역_점수_등급.py",
    "5_백테스트_성적표.py",
    "6_단지_시세.py",
}  # 실거래·점수 화면: 투자 권유 아님 문구
PUBLIC_DIR = Path(__file__).resolve().parents[1] / "data" / "public"


@pytest.fixture(autouse=True)
def _local_data(monkeypatch):
    """테스트는 TiDB 대신 로컬 Parquet 으로 (Secrets 가 있어도)."""
    monkeypatch.setenv("REALESTATE_DATA", "local")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_has_footer(page):
    src = page.read_text(encoding="utf-8")
    calls = re.findall(r"common\.footer\(([^)]*)\)", src)
    assert calls, f"{page.name}: common.footer() 호출이 없습니다"
    if page.name in DISCLAIMER_PAGES:
        assert all("disclaimer=True" in c for c in calls), (
            f"{page.name}: 면책 문구(disclaimer=True)가 빠졌습니다"
        )
    if page.name in MVP3_PAGES:
        assert all("invest_note=True" in c for c in calls), f"{page.name}: 투자 권유 아님 문구가 빠졌습니다"


def test_footer_always_has_rone_source():
    import common

    assert "reb.or.kr/r-one" in common.SOURCE_TEXT["rone"]
    assert common.DISCLAIMER_SHORT == "공공 자료를 가공한 것이며 투자 조언이 아닙니다"


def test_required_notes_text():
    import common

    assert common.INVEST_NOTE == "투자 권유가 아닙니다. 공공 데이터 기반 분석 결과입니다."
    assert common.REGION_NOTE == (
        "이 점수는 권역(수도권/지방) 흐름을 반영합니다. 같은 권역 안 지역 비교 근거로 쓰기 어렵습니다"
        "(평가 기간 권역 안 IC가 0과 구분되지 않음)."
    )
    assert (
        common.rule_text("매매 절대 불가", "R2,R3")
        == "R2 거래 급감 + 하락 지속 (검증 안 됨) · R3 해제 급증 (검증 안 됨)"
    )
    assert common.rule_text("데이터 부족", "") == "거래가 적어 판단하지 않음"
    assert common.rule_text("매매 추천", "") == ""


@pytest.mark.skipif(not (RONE_CLEAN / "rone_index.parquet").exists(), reason="build_rone.py 를 먼저 실행")
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_runs_without_error(page):
    if page.name in MVP3_PAGES and not (PUBLIC_DIR / "region_score.parquet").exists():
        pytest.skip("build_public.py 를 먼저 실행")
    if page.name == "6_단지_시세.py" and not (PUBLIC_DIR / "complex_price.parquet").exists():
        pytest.skip("build_complex_price.py 를 먼저 실행")
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception, at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "reb.or.kr/r-one" in captions
    if page.name in MVP3_PAGES:
        assert captions.rstrip().endswith("투자 권유가 아닙니다. 공공 데이터 기반 분석 결과입니다.")
    shown = " ".join(str(e.value) for e in [*at.warning, *at.markdown, *at.caption])
    if page.name == "3_실거래_집계.py":
        info = " ".join(str(e.value) for e in at.info)
        assert "최근 2개월(2026.9·2026.10)" in info and "신고 기한(계약 후 30일)" in info
    if page.name in {"4_지역_점수_등급.py", "5_백테스트_성적표.py"}:
        assert "권역 안 IC가 0과 구분되지 않음" in shown
        assert "검증 안 됨" in shown
    if page.name in {"3_실거래_집계.py", "4_지역_점수_등급.py", "5_백테스트_성적표.py", "6_단지_시세.py"}:
        assert "이 화면에서 볼 수 있는 것" in shown
    if page.name == "6_단지_시세.py":
        assert "단지 단위는 검증하지 않은 참고 정보입니다." in shown
        assert "추천" not in " ".join(str(e.value) for e in at.markdown)


def test_complex_page_never_says_recommend():
    src = (APP / "pages" / "6_단지_시세.py").read_text(encoding="utf-8")
    assert "추천" not in src


def test_grade_colors_cover_all_grades():
    import common

    assert set(common.GRADE_COLORS) == set(common.GRADES)
    assert common.grade_style("매매 절대 불가").startswith("background-color: #d63c3c")
