import pandas as pd
import streamlit as st

import common

st.set_page_config(page_title="지역 점수·등급", page_icon="🏷️", layout="wide")
st.title("지역 점수·등급")

score, backend = common.load_public("region_score")
names, _ = common.load_public("region_name")
df = score.merge(names, on="sgg_cd", how="left")
asof = df["asof_ym"].max()

st.write(
    f"**{common.fmt_ym(asof)} 초 기준** 시군구별 매매 점수와 등급입니다. "
    "가격이 내려온 정도를 4요소(고점 대비 하락률, 모델 예측가 대비 저평가, 최근 3개월 추세, 거래량 변화)로 매겨 "
    "달마다 전국 백분위로 바꾸고 같은 비중(각 25%)으로 더했습니다."
)
st.warning(common.REGION_NOTE, icon="⚠️")

GRADES = ["매매 추천", "매매 고려", "매매 고려 대상 아님", "매매 절대 불가", "데이터 부족"]
c1, c2 = st.columns([3, 2])
sidos = sorted(df["sido_nm"].dropna().unique())
pick_sido = c1.multiselect("시도", sidos, placeholder="전체 시도")
pick_grade = c2.multiselect("등급", GRADES, placeholder="전체 등급")
view = df
if pick_sido:
    view = view[view["sido_nm"].isin(pick_sido)]
if pick_grade:
    view = view[view["grade"].isin(pick_grade)]

counts = view["grade"].value_counts()
cols = st.columns(len(GRADES))
for col, g in zip(cols, GRADES, strict=True):
    help_ = {
        "매매 절대 불가": f"규칙 R2·R3 중 하나에 해당 ({common.UNVERIFIED}: 평가 기간에 R2 0건, R3 정답 3건)",
        "데이터 부족": f"{common.NODATA_NOTE} — 등급이 아님 (12개월 거래 30건 미만이거나 3개월 중개거래 10건 미만)",
    }.get(g)
    col.metric(g, f"{int(counts.get(g, 0)):,}곳", help=help_)

order = {g: i for i, g in enumerate(GRADES)}
table = view.assign(
    _o=view["grade"].map(order),
    규칙=[common.rule_text(g, f) for g, f in zip(view["grade"], view["rule_flags"], strict=True)],
).sort_values(["_o", "score"], ascending=[True, False])
out = pd.DataFrame(
    {
        "시도": table["sido_nm"],
        "시군구": table["sgg_nm"],
        "등급": table["grade"],
        "규칙·안내": table["규칙"],
        "점수": table["score"],
        "A 고점 대비 하락": table["p_a"],
        "B 모델 저평가": table["p_b"],
        "C 3개월 추세": table["p_c"],
        "D 거래량 변화": table["p_d"],
        "고점 대비 (%)": -table["drawdown_pct"],
        "3개월 지수 변화 (%)": table["trend_3m_pct"],
        "저평가 (%)": table["underval_pct"],
        "거래량 (전년 대비 배)": table["volume_ratio"],
        "12개월 거래": table["trade_12m"],
    }
)
pct = {"min_value": 0, "max_value": 100, "format": "%.0f"}
st.dataframe(
    out,
    hide_index=True,
    width="stretch",
    height=560,
    column_config={
        "점수": st.column_config.ProgressColumn("점수", help="4요소 백분위의 평균 (0~100)", **pct),
        "A 고점 대비 하락": st.column_config.NumberColumn(
            help="전국 백분위. 최근 5년 고점 대비 많이 내렸을수록 높음", format="%.0f"
        ),
        "B 모델 저평가": st.column_config.NumberColumn(
            help="전국 백분위. 최근 3개월 중개거래가 모델 시세보다 쌀수록 높음", format="%.0f"
        ),
        "C 3개월 추세": st.column_config.NumberColumn(
            help="전국 백분위. 최근 3개월 조사 지수가 오를수록 높음", format="%.0f"
        ),
        "D 거래량 변화": st.column_config.NumberColumn(
            help="전국 백분위. 거래량이 전년 같은 기간보다 늘수록 높음", format="%.0f"
        ),
        "고점 대비 (%)": st.column_config.NumberColumn(format="%.1f"),
        "3개월 지수 변화 (%)": st.column_config.NumberColumn(format="%.2f"),
        "저평가 (%)": st.column_config.NumberColumn(
            help="+면 모델 시세보다 싸게 거래됨 (건수로 0 쪽으로 당긴 값)", format="%.2f"
        ),
        "거래량 (전년 대비 배)": st.column_config.NumberColumn(format="%.2f"),
    },
)
st.caption(
    f"{len(out):,}곳 · 데이터 부족은 등급이 아니며 점수 칸이 비어 있을 수 있음 · "
    f"A·C 는 R-ONE 조사 지수 (시군구 지수가 없는 {int(df['sido_level_index'].sum())}곳은 시도 값) · 자료 위치: {backend}"
)

with st.expander("등급은 어떻게 정하나요"):
    st.markdown(
        f"""
| 순서 | 조건 | 등급 |
|---|---|---|
| 1 | 규칙 R2 또는 R3 에 해당 | **매매 절대 불가** |
| 2 | 규칙 R1 에 해당 | **데이터 부족** — {common.NODATA_NOTE} (등급 아님) |
| 3 | 점수 ≥ 70 이고 C 3개월 추세 백분위 ≥ 40 | 매매 추천 |
| 4 | 점수 50 이상 (또는 70 이상인데 추세 < 40) | 매매 고려 |
| 5 | 나머지 | 매매 고려 대상 아님 |

**규칙** (점수만으로는 절대 불가가 붙지 않음)
- {common.RULE_LABEL["R2"]}: 거래량이 전년 같은 기간의 절반 이하 **그리고** 조사 지수 3개월 연속 하락 **그리고** 3개월 −3% 이하
- {common.RULE_LABEL["R3"]}: 최근 3개월 거래 중 그때까지 해제된 비율 > 15%
- R1 데이터 부족: 최근 12개월 정상 거래 30건 미만, 또는 최근 3개월 중개거래 10건 미만 (B 를 못 구함)

R2 는 평가 기간(2024.1~2025.9)에 하락기가 없어 한 번도 걸리지 않았고, R3 는 정답이 있는 사례가 3건뿐이라 둘 다 효과가 검증되지 않았습니다.
점수·등급 설정: `{df["model_version"].iloc[0]}`. 자세한 성적은 **백테스트 성적표** 페이지.
"""
    )

common.footer(sources=["molit_onbid"], disclaimer=True, contract_basis=True, invest_note=True)
