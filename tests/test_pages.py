"""화면 규칙 검사: 모든 페이지에 출처 푸터, 순위·점수 페이지에 면책 문구 (PLAN.md 10장)."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from realestate.config import RONE_CLEAN

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]
DISCLAIMER_PAGES = {"2_하락_지역_순위.py"}


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_page_has_footer(page):
    src = page.read_text(encoding="utf-8")
    calls = re.findall(r"common\.footer\(([^)]*)\)", src)
    assert calls, f"{page.name}: common.footer() 호출이 없습니다"
    if page.name in DISCLAIMER_PAGES:
        assert all("disclaimer=True" in c for c in calls), (
            f"{page.name}: 면책 문구(disclaimer=True)가 빠졌습니다"
        )


def test_footer_always_has_rone_source():
    import common

    assert "reb.or.kr/r-one" in common.SOURCE_TEXT["rone"]
    assert common.DISCLAIMER_SHORT == "공공 자료를 가공한 것이며 투자 조언이 아닙니다"


@pytest.mark.skipif(not (RONE_CLEAN / "rone_index.parquet").exists(), reason="build_rone.py 를 먼저 실행")
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_runs_without_error(page):
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception, at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "reb.or.kr/r-one" in captions
