"""Streamlit 페이지 공통: 데이터 읽기, 출처·안내 푸터, 차트 색.

데이터 읽기 순서
1. Streamlit Secrets 에 [tidb] 가 있으면 TiDB(읽기 전용 계정)에서 읽는다 → 공개 배포용
2. 없으면 로컬 data/clean/rone/*.parquet 을 읽는다 → 로컬 개발용
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # Streamlit Cloud 에서는 패키지 설치 없이 src 를 바로 쓴다
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from realestate.config import RONE_CLEAN  # noqa: E402

SOURCE_TEXT = {
    "rone": "자료: 한국부동산원 R-ONE 부동산통계정보 (reb.or.kr/r-one)",
    "molit_onbid": "자료: 국토교통부 실거래가 공개시스템, 한국자산관리공사 온비드",
}
DISCLAIMER_SHORT = "공공 자료를 가공한 것이며 투자 조언이 아닙니다"
CONTRACT_BASIS_NOTE = "실거래 집계는 계약일 기준 참고치이며 공식통계가 아닙니다"
DISCLAIMER_FULL = (
    "이 서비스는 개인 포트폴리오용 데모이며 투자 조언이 아닙니다. "
    "지표와 등급은 공공데이터와 통계 모델로 만든 참고 자료이고, 정확성을 보장하지 않습니다. "
    "실제 거래·입찰 전에 반드시 등기부, 현장, 전문가 확인을 하세요."
)
INDEX_LABEL = {"survey": "매매가격지수(조사)", "real": "실거래가격지수"}

# 범주형 색 (고정 순서, 순위가 아니라 지역에 붙는다). dataviz 기본 팔레트, 라이트/다크 각각 검증된 값
_SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
}
MAX_SERIES = len(_SERIES["light"])


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001  테마 정보를 못 얻으면 라이트로
        return "light"


def series_colors() -> list[str]:
    return _SERIES[theme_mode()]


def _tidb_secrets() -> dict | None:
    try:
        if "tidb" in st.secrets:
            return dict(st.secrets["tidb"])
    except Exception:  # noqa: BLE001  secrets.toml 이 아예 없을 때
        return None
    return None


@st.cache_data(ttl=3600, show_spinner="데이터 불러오는 중…")
def load_rone() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(지수, 지역표, 데이터 위치 설명)."""
    secrets = _tidb_secrets()
    if secrets:
        from realestate.db import tidb

        engine = tidb.make_engine(
            secrets["host"],
            int(secrets.get("port", 4000)),
            secrets["user"],
            secrets["password"],
            secrets["database"],
        )
        tidb.assert_read_only(engine)
        index = pd.read_sql("SELECT source, region_key, ym, index_value, mom_pct FROM rone_index", engine)
        region = pd.read_sql("SELECT * FROM rone_region", engine)
        for df, cols in ((index, ["ym"]), (region, ["first_ym", "last_ym"])):
            for c in cols:
                df[c] = pd.to_datetime(df[c])
        region["has_real"] = region["has_real"].astype(bool)
        backend = "TiDB (읽기 전용)"
    else:
        index_path, region_path = RONE_CLEAN / "rone_index.parquet", RONE_CLEAN / "rone_region.parquet"
        if not index_path.exists():
            st.error("정리된 데이터가 없습니다. 먼저 `uv run python scripts/build_rone.py` 를 실행하세요.")
            st.stop()
        index, region = pd.read_parquet(index_path), pd.read_parquet(region_path)
        backend = "로컬 Parquet"
    return index, region, backend


def banner() -> None:
    st.info(DISCLAIMER_FULL, icon="ℹ️")


def footer(sources: tuple[str, ...] = (), disclaimer: bool = False, contract_basis: bool = False) -> None:
    """모든 페이지 맨 아래에 호출한다. R-ONE 출처는 항상 붙는다 (PLAN.md 10장)."""
    lines = [SOURCE_TEXT["rone"]]
    lines += [SOURCE_TEXT[s] for s in sources if s != "rone"]
    if contract_basis:
        lines.append(CONTRACT_BASIS_NOTE)
    if disclaimer:
        lines.append(DISCLAIMER_SHORT)
    st.divider()
    st.caption("  \n".join(lines))


def fmt_ym(ts: pd.Timestamp) -> str:
    return f"{ts.year}년 {ts.month}월"
