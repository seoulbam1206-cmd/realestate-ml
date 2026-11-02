import pandas as pd
import streamlit as st

import common

st.set_page_config(page_title="단지 시세", page_icon="🏢", layout="wide")
st.title("단지 시세")

COMPLEX_NOTE = (
    "단지 단위는 검증하지 않은 참고 정보입니다. "
    "모델보다 낮게 거래된 곳이 이후 더 오른다는 근거는 없습니다(사후 분석 참고)."
)
SIMILAR_PCT = 2.0  # |차이| 가 이보다 작으면 "모델과 비슷" (거래 1건 모델 오차 중위값 약 4%)

cp, backend = common.load_public("complex_price")
names, _ = common.load_public("region_name")
w0, w1 = cp["window_from"].min(), cp["window_to"].max()
span = f"{w0.year}.{w0.month}~{w1.year}.{w1.month}"

common.intro(
    [
        "시도 → 시군구를 고르면 그 안 아파트 단지의 **최근 12개월 실거래 ㎡당 중위가**와 **모델 적정가**를 나란히 봅니다.",
        f"계약월 {span} (모델 예측이 있는 마지막 12개월), 지금 기준 해제되지 않은 거래만 셉니다. 등급은 매기지 않습니다.",
        "차이(%)는 실제 거래가가 모델 적정가보다 얼마나 높았는지(+) 낮았는지(−)입니다.",
    ]
)
st.warning(COMPLEX_NOTE, icon="⚠️")

c1, c2, c3 = st.columns([1.3, 1.6, 2])
sidos = sorted(names["sido_nm"].unique())
sido = c1.selectbox("시도", sidos, index=sidos.index("서울특별시") if "서울특별시" in sidos else 0)
in_sido = names[names["sido_nm"] == sido].sort_values("sgg_nm")
vol = cp.groupby("sgg_cd")["trade_cnt"].sum()
in_sido = in_sido[in_sido["sgg_cd"].isin(vol.index)]
sgg_name = in_sido.set_index("sgg_cd")["sgg_nm"]
sgg = c2.selectbox("시군구", sgg_name.index.tolist(), format_func=lambda c: sgg_name[c])
query = c3.text_input("단지명 검색", placeholder="일부만 입력해도 됨 (공백 무시)")

view = cp[cp["sgg_cd"] == sgg]
if query:
    flat = view["apt_nm"].str.replace(" ", "", regex=False)
    view = view[flat.str.contains(query.replace(" ", ""), case=False, regex=False)]
only_priced = st.toggle(
    "거래 5건 이상만 보기", value=True, help="끄면 거래 5건 미만 단지도 보임 (가격 칸은 비어 있음)"
)
if only_priced:
    view = view[view["median_ppm2"].notna()]


def direction(gap: float) -> str:
    if pd.isna(gap):
        return "거래 5건 미만"
    if abs(gap) < SIMILAR_PCT:
        return "모델과 비슷"
    return "모델 대비 높게 거래됨" if gap > 0 else "모델 대비 낮게 거래됨"


DIR_COLORS = {  # 방향만 구분하는 중립색 (좋고 나쁨이 아님)
    "모델 대비 높게 거래됨": "color: #c2410c",
    "모델 대비 낮게 거래됨": "color: #1d4ed8",
}
view = view.sort_values("trade_cnt", ascending=False)
out = pd.DataFrame(
    {
        "단지명": view["apt_nm"],
        "법정동": view["umd_nm"],
        "건축년도": view["build_year"],
        "최근 12개월 거래": view["trade_cnt"],
        "실거래 ㎡당 중위가 (만 원)": view["median_ppm2"].round(0),
        "모델 적정가 ㎡당 (만 원)": view["model_ppm2"].round(0),
        "차이 (%)": view["gap_pct"].round(1),
        "모델 대비": view["gap_pct"].map(direction),
    }
)
k1, k2, k3 = st.columns(3)
priced = out["차이 (%)"].notna()
k1.metric("단지 수", f"{len(out):,}곳", help="이 시군구에서 최근 12개월 거래가 있는 단지 (위 조건 적용)")
k2.metric("거래 수 합", f"{int(out['최근 12개월 거래'].sum()):,}건")
k3.metric(
    "차이 중위값",
    "-" if not priced.any() else f"{out.loc[priced, '차이 (%)'].median():+.1f}%",
    help="가격이 있는 단지들의 차이(%) 가운데 값. 시군구 전체가 모델보다 높게/낮게 거래되는지 대략 보여 줌",
)
st.dataframe(
    out.style.map(lambda v: DIR_COLORS.get(v, ""), subset=["모델 대비"]),
    hide_index=True,
    width="stretch",
    height=560,
    column_config={
        "건축년도": st.column_config.NumberColumn(
            format="%d", help="같은 동에 이름이 같은 단지를 구분하는 데 씀"
        ),
        "최근 12개월 거래": st.column_config.NumberColumn(format="localized"),
        "실거래 ㎡당 중위가 (만 원)": st.column_config.NumberColumn(
            format="localized", help=common.TERMS["중위가"]
        ),
        "모델 적정가 ㎡당 (만 원)": st.column_config.NumberColumn(
            format="localized", help=common.TERMS["모델 적정가"] + ". 같은 거래들의 모델 값 ㎡당 중위값"
        ),
        "차이 (%)": st.column_config.NumberColumn(
            format="%+.1f",
            help="거래마다 (실거래가 ÷ 모델 적정가 − 1) 의 중위값. 두 ㎡당 중위가를 나눈 값과 조금 다를 수 있음",
        ),
    },
)
st.caption(
    f"{common.term('모델 적정가')} · 차이가 ±{SIMILAR_PCT:.0f}% 안이면 '모델과 비슷' "
    "(거래 1건 기준 모델 오차가 보통 4% 안팎이라 작은 차이는 의미가 적음) · "
    f"거래 5건 미만 단지는 가격을 비움 · 개별 거래는 보여 주지 않음 · 자료 위치: {backend}"
)

with st.expander("이 숫자를 어떻게 봐야 하나요"):
    st.markdown(
        f"""
- **모델 적정가**는 시세 모델(LightGBM, 고정 설정 `config/lgbm.json`)이 단지·면적·층·연식·최근 주변 시세로 추정한 가격입니다.
  거래마다 그 거래를 학습에 쓰지 않은 모델(분기마다 다시 학습)의 값을 썼습니다.
- 시세 모델의 거래 1건 오차(MAPE)는 평가 기간 약 6% 입니다. 단지마다 층·향·동·수리 상태 같은 모델이 모르는 차이가 있어,
  한 단지가 꾸준히 높게/낮게 나오는 것은 단지 특성일 수 있습니다.
- 지역 점수의 사후 분석(백테스트 성적표)에서 **"모델보다 싸게 거래된 곳(B)"은 이후 오히려 덜 오른** 경향이었습니다
  (평가 6개월 "B 만" IC 음수). 그래서 이 화면은 "싸다/비싸다" 판단이 아니라 정보로만 보여 줍니다.
- {common.INVEST_NOTE}
"""
    )

common.footer(sources=["molit_onbid"], disclaimer=True, contract_basis=True, invest_note=True)
