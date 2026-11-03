import math

import pandas as pd
import streamlit as st

import common
from realestate.models import serve
from realestate.score import explain

st.set_page_config(page_title="호가 비교", page_icon="🧮", layout="wide")
st.title("호가 비교")

ASK_NOTE = serve.ASK_NOTE
JUDGE_COLORS = {  # 방향만 구분하는 중립색 (좋고 나쁨이 아님, 단지 시세 화면과 같은 색)
    "모델 대비 높음": "#c2410c",
    "모델 대비 낮음": "#1d4ed8",
    "모델과 비슷": "#4a4f57",
}


def manwon(v: float) -> str:
    """만 원 → '24억 5,000만 원'."""
    v = int(round(v))
    eok, man = divmod(v, 10000)
    if eok and man:
        return f"{eok}억 {man:,}만 원"
    return f"{eok}억 원" if eok else f"{man:,}만 원"


names, _ = common.load_public("region_name")
common.intro(
    [
        "단지·전용면적·층과 **호가**를 넣으면 시세 모델의 **적정가**와 비교해 차이(%)와 판정을 보여 줍니다.",
        "판정은 평가 기간(2024.1~2025.9)에 실제 거래가가 모델 값에서 벗어난 폭(80% 범위)을 기준으로 합니다. "
        "범위 안이면 '모델과 비슷'입니다.",
        "아래에 그 시군구의 지역 등급(시군구 단위)도 함께 보여 줍니다. 단지에 매긴 등급이 아닙니다.",
    ]
)
st.warning(ASK_NOTE, icon="⚠️")

c1, c2, c3 = st.columns([1.2, 1.4, 2.4])
sidos = sorted(names["sido_nm"].unique())
sido = c1.selectbox("시도", sidos, index=sidos.index("서울특별시") if "서울특별시" in sidos else 0)
in_sido = names[names["sido_nm"] == sido].sort_values("sgg_nm")
sgg_name = in_sido.set_index("sgg_cd")["sgg_nm"]
sgg = c2.selectbox("시군구", sgg_name.index.tolist(), format_func=lambda c: sgg_name[c])

cx, backend = common.load_fair(sgg_cd=sgg)
if cx.empty:
    st.info("이 시군구에는 적정가 표에 있는 단지가 없습니다 (최근 36개월 거래가 있는 단지만 있음).")
    common.footer(sources=["molit_onbid"], disclaimer=True, contract_basis=True, invest_note=True)
    st.stop()
cx = cx.sort_values(["apt_nm", "umd_nm", "build_year"])
label = {
    r.complex_id: f"{r.apt_nm} ({r.umd_nm}"
    + ("" if pd.isna(r.build_year) else f", {int(r.build_year)}년")
    + ")"
    for r in cx.itertuples()
}
cid = c3.selectbox(
    "단지",
    list(label),
    format_func=label.get,
    help="이름 일부를 입력하면 걸러짐. 같은 동에 이름이 같은 단지는 건축년도로 구분",
)

rows, _ = common.load_fair(complex_id=cid)
areas = sorted(rows["area_m2"].unique())
d1, d2, d3 = st.columns([1.2, 1, 1.6])
area = d1.selectbox(
    "전용면적 (㎡)",
    areas,
    index=min(range(len(areas)), key=lambda i: abs(areas[i] - 84.9)),
    format_func=lambda a: f"{a:.2f}㎡ (약 {a / 3.3058:.0f}평)",
    help="이 단지에서 최근 36개월 거래된 전용면적",
)
at_area = rows[rows["area_m2"] == area]
top = int(at_area["floor_rep"].max()) if len(at_area) else 1
floor = d2.number_input("층", min_value=1, max_value=99, value=min(10, max(top, 1)), step=1)
ask = d3.number_input("호가 (만 원)", min_value=0, value=0, step=500, help="0 이면 적정가만 보여 줌")

band = serve.floor_bucket(int(floor))
row = at_area[at_area["floor_band"] == band]
moved = row.empty
if moved:  # 이 단지 최고층보다 높은 층을 넣은 경우 → 가장 높은 구간
    row = at_area.sort_values("floor_rep").tail(1)
r = row.iloc[0]

basis = serve.basis_text(pd.Timestamp(r["target_ym"]))  # "2026년 10월 기준 적정가"
st.subheader(f"모델 적정가 · {basis}")
k1, k2, k3 = st.columns(3)
k1.metric(
    basis,
    manwon(r["fair_price"]),
    help=common.TERMS["모델 적정가"]
    + f". 층은 '{r['floor_band']}' 구간 대표 {int(r['floor_rep'])}층으로 계산",
)
k2.metric("㎡당", f"{r['fair_ppm2']:,.0f}만 원")
k3.metric(
    "80% 범위",
    f"{manwon(r['range_low'])} ~ {manwon(r['range_high'])}",
    help="평가 기간에 실제 거래가가 모델 값 기준 이 범위 안에 든 비율이 80%",
)
st.caption(
    f"층 구간 '{r['floor_band']}' (대표 {int(r['floor_rep'])}층)"
    + (
        f" — 입력한 {int(floor)}층은 이 단지에서 거래된 최고층보다 높아 가장 높은 구간으로 계산"
        if moved
        else ""
    )
    + f" · 이 단지·면적대 최근 12개월 거래 **{r['hist_group']}** 구간의 오차 범위"
    + f" · {basis}: 그 달에 거래된다고 보고 계산한 값"
    + f" (모델 기준일 {pd.Timestamp(r['model_anchor']):%Y.%m.%d}, 그때까지 공개된 거래만 사용)"
)

if ask > 0:
    j = serve.judge(ask, r["fair_price"], r["range_low"], r["range_high"], r["hist_group"])
    color = JUDGE_COLORS[j["judgement"]]
    note = (
        f' <span style="color:#b45309;font-weight:600">· {j["judgement_note"]}</span>'
        if j["judgement_note"]
        else ""
    )
    st.subheader("호가 비교")
    m1, m2 = st.columns([1, 2])
    m1.metric("호가와의 차이", f"{j['ask_gap_pct']:+.1f}%", help="호가 ÷ 모델 적정가 − 1")
    m2.markdown(
        f'<div style="font-size:0.9rem;opacity:0.7">판정</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:{color}">{j["judgement"]}{note}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"호가 {manwon(ask)} vs 80% 범위 {manwon(r['range_low'])} ~ {manwon(r['range_high'])} · "
        "범위 안이면 '모델과 비슷'. 단지 시세 화면의 ±2% 기준과 다른 이유: 그 화면은 여러 거래의 중위값, 여기는 한 건"
    )
    if j["judgement_note"]:
        st.info(
            f"{j['judgement_note']}: 이 단지·면적대는 최근 12개월 거래가 없어, 평가 기간에도 오차가 "
            "다른 구간의 2~3배였습니다 (80% 범위 약 −22% ~ +25%).",
            icon="ℹ️",
        )

st.subheader(explain.REGION_LABEL)
score, _ = common.load_public("region_score")
reg = score[score["sgg_cd"] == sgg]
if reg.empty:
    st.write("이 시군구는 지역 점수가 없습니다.")
else:
    g = reg.iloc[0]
    sc = "" if pd.isna(g["score"]) else f" · 점수 {g['score']:.0f}"
    extra = common.rule_text(g["grade"], g["rule_flags"])
    st.markdown(
        f"{sgg_name[sgg]}: {common.grade_badge(g['grade'])}{sc}"
        + (f" · {extra}" if extra else "")
        + f" ({common.fmt_ym(g['asof_ym'])} 초 기준)",
        unsafe_allow_html=True,
    )
    st.caption(
        "시군구 전체 흐름으로 매긴 등급이며 이 단지·호가에 대한 판단이 아닙니다. "
        "자세한 이유는 **지역 점수·등급** 화면."
    )
st.warning(common.REGION_NOTE, icon="⚠️")

with st.expander("이 숫자를 어떻게 봐야 하나요"):
    bands = serve.load_error_band()["groups"]
    tbl = pd.DataFrame(
        [
            {
                "최근 12개월 거래 (단지·면적대)": k,
                "평가 기간 거래 수": v["n"],
                "오차 중위값 (%)": v["mdape_pct"],
                "80% 범위": f"{math.expm1(v['q10']) * 100:+.1f}% ~ {math.expm1(v['q90']) * 100:+.1f}%",
            }
            for k, v in bands.items()
            if k != "전체"
        ]
    )
    st.dataframe(tbl, hide_index=True)
    st.markdown(
        f"""
- **모델 적정가**: 시세 모델(LightGBM, 고정 설정 `config/lgbm.json`)을 기준일 2026.9.30 까지 공개된 거래로 한 번 학습해,
  그 단지·면적·층·연식·최근 주변 시세로 계산한 값입니다. 이 기준일 모델 자체는 평가된 적이 없고,
  오차 범위는 같은 설정의 평가 기간(2024.1~2025.9, 분기마다 다시 학습) 실제 오차를 빌려 씁니다.
- 거래 구분(중개거래·개인 간 거래)은 가장 흔한 경우로 가정했습니다. 층·향·동·수리 상태 같은 모델이 모르는 차이는 반영되지 않습니다.
- 층은 5개 구간(1~3 · 4~7 · 8~12 · 13~20 · 21층 이상)의 대표 층으로 미리 계산한 값입니다.
- {common.INVEST_NOTE}
"""
    )
st.caption(f"자료 위치: {backend}")

common.footer(sources=["molit_onbid"], disclaimer=True, contract_basis=True, invest_note=True)
