"""지역 등급 한 줄 사유 (화면 '지역 점수·등급' 과 API /region 이 같이 씀). 등급 규칙은 바꾸지 않고 설명만."""

from __future__ import annotations

import pandas as pd

FACTORS = {"p_a": "고점 대비 하락", "p_b": "모델 저평가", "p_c": "3개월 추세", "p_d": "거래량 변화"}
NODATA_NOTE = "거래가 적어 판단하지 않음"
UNVERIFIED = "검증 안 됨"
REGION_NOTE = (
    "이 점수는 권역(수도권/지방) 흐름을 반영합니다. 같은 권역 안 지역 비교 근거로 쓰기 어렵습니다"
    "(평가 기간 권역 안 IC가 0과 구분되지 않음)."
)
REGION_LABEL = "지역 등급(시군구 단위)"
RULE_LABEL = {"R2": f"R2 거래 급감 + 하락 지속 ({UNVERIFIED})", "R3": f"R3 해제 급증 ({UNVERIFIED})"}


def rule_text(grade: str, flags: str) -> str:
    """절대 불가면 걸린 규칙 이름 (검증 안 됨), 데이터 부족이면 안내, 그 밖에는 빈 문자열."""
    if grade == "데이터 부족":
        return NODATA_NOTE
    if grade == "매매 절대 불가":
        return " · ".join(RULE_LABEL[f] for f in (flags or "").split(",") if f)
    return ""


def reason(r: pd.Series) -> str:
    """등급이 나온 이유 한 줄 (등급 규칙 그대로 + 4요소 백분위 중 가장 높은·낮은 것)."""
    g = r["grade"]
    if g in ("매매 절대 불가", "데이터 부족"):
        return rule_text(g, r["rule_flags"])
    s, pc = r["score"], r["p_c"]
    if g == "매매 추천":
        head = f"점수 {s:.0f} (70 이상), 추세 백분위 {pc:.0f} (40 이상)"
    elif g == "매매 고려" and s >= 70:
        head = f"점수 {s:.0f} 이지만 추세 백분위 {pc:.0f} (40 미만)"
    elif g == "매매 고려":
        head = f"점수 {s:.0f} (50~70)"
    else:
        head = f"점수 {s:.0f} (50 미만)"
    p = r[list(FACTORS)].astype(float)
    hi, lo = p.idxmax(), p.idxmin()
    return f"{head} · 높음: {FACTORS[hi]} {p[hi]:.0f} · 낮음: {FACTORS[lo]} {p[lo]:.0f}"
