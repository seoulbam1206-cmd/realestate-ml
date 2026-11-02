"""Streamlit 페이지 공통: 데이터 읽기, 출처·안내 푸터, 차트 색.

데이터 읽기 순서
1. Streamlit Secrets 에 [tidb] 가 있으면 TiDB(읽기 전용 계정 web_ro)에서 읽는다 → 공개 배포용
2. 없으면 로컬 Parquet (R-ONE: data/clean/rone, MVP 3 표: data/public) → 로컬 개발용
   환경 변수 REALESTATE_DATA=local 이면 Secrets 가 있어도 로컬 (테스트·업로드 전 확인용)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # Streamlit Cloud 에서는 패키지 설치 없이 src 를 바로 쓴다
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from realestate.config import RONE_CLEAN, ROOT  # noqa: E402

PUBLIC_DIR = ROOT / "data" / "public"

SOURCE_TEXT = {
    "rone": "자료: 한국부동산원 R-ONE 부동산통계정보 (reb.or.kr/r-one)",
    "molit_onbid": "자료: 국토교통부 실거래가 공개시스템, 한국자산관리공사 온비드",
}
DISCLAIMER_SHORT = "공공 자료를 가공한 것이며 투자 조언이 아닙니다"
CONTRACT_BASIS_NOTE = "실거래 집계는 계약일 기준 참고치이며 공식통계가 아닙니다"
INVEST_NOTE = "투자 권유가 아닙니다. 공공 데이터 기반 분석 결과입니다."
REGION_NOTE = (
    "이 점수는 권역(수도권/지방) 흐름을 반영합니다. 같은 권역 안 지역 비교 근거로 쓰기 어렵습니다"
    "(평가 기간 권역 안 IC가 0과 구분되지 않음)."
)
UNVERIFIED = "검증 안 됨"
PROVISIONAL_NOTE = "신고 기한(계약 후 30일) 때문에 거래 수가 아직 덜 잡혔을 수 있음"
PROVISIONAL_MONTHS = 2  # 실거래 집계에서 잠정으로 표시할 최근 달 수
NODATA_NOTE = "거래가 적어 판단하지 않음"
RULE_LABEL = {"R2": f"R2 거래 급감 + 하락 지속 ({UNVERIFIED})", "R3": f"R3 해제 급증 ({UNVERIFIED})"}
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
    if os.environ.get("REALESTATE_DATA") == "local":
        return None
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


@st.cache_resource(show_spinner=False)
def _ro_engine():
    """TiDB 읽기 전용 엔진 (한 번 만들고 재사용). 쓰기 권한이 있는 계정이면 멈춤."""
    from realestate.db import tidb

    s = _tidb_secrets()
    engine = tidb.make_engine(s["host"], int(s.get("port", 4000)), s["user"], s["password"], s["database"])
    tidb.assert_read_only(engine)
    return engine


@st.cache_data(ttl=3600, show_spinner="데이터 불러오는 중…")
def load_public(name: str) -> tuple[pd.DataFrame, str]:
    """MVP 3 공개 집계표 하나 (scripts/build_public.py → publish.py). (표, 데이터 위치)."""
    if name not in PUBLIC_TABLES:
        raise ValueError(name)
    if _tidb_secrets():
        df = pd.read_sql(f"SELECT * FROM {name}", _ro_engine())
        backend = "TiDB (읽기 전용)"
    else:
        path = PUBLIC_DIR / f"{name}.parquet"
        if not path.exists():
            st.error("공개 집계표가 없습니다. 먼저 `uv run python scripts/build_public.py` 를 실행하세요.")
            st.stop()
        df = pd.read_parquet(path)
        backend = "로컬 Parquet"
    for c in ("ym", "asof_ym"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    for c in ("r1", "r2", "r3", "sido_level_index"):
        if c in df:
            df[c] = df[c].astype(bool)
    return df, backend


PUBLIC_TABLES = (
    "region_name",
    "agg_region_month",
    "region_score",
    "bt_ic",
    "bt_grade",
    "bt_baseline",
    "bt_meta",
)


def rule_text(grade: str, flags: str) -> str:
    """등급 옆에 붙는 규칙 설명: 절대 불가면 걸린 규칙 이름 (검증 안 됨), 데이터 부족이면 안내."""
    if grade == "데이터 부족":
        return NODATA_NOTE
    if grade == "매매 절대 불가":
        return " · ".join(RULE_LABEL[f] for f in flags.split(",") if f)
    return ""


def banner() -> None:
    st.info(DISCLAIMER_FULL, icon="ℹ️")


def footer(
    sources: tuple[str, ...] = (),
    disclaimer: bool = False,
    contract_basis: bool = False,
    invest_note: bool = False,
) -> None:
    """모든 페이지 맨 아래에 호출한다. R-ONE 출처는 항상 붙는다 (PLAN.md 10장). invest_note 는 맨 마지막 줄."""
    lines = [SOURCE_TEXT["rone"]]
    lines += [SOURCE_TEXT[s] for s in sources if s != "rone"]
    if contract_basis:
        lines.append(CONTRACT_BASIS_NOTE)
    if disclaimer:
        lines.append(DISCLAIMER_SHORT)
    if invest_note:
        lines.append(INVEST_NOTE)
    st.divider()
    st.caption("  \n".join(lines))


def fmt_ym(ts: pd.Timestamp) -> str:
    return f"{ts.year}년 {ts.month}월"
