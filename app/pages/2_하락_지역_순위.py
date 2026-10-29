import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
from realestate.ingest.rone import REGION_TYPES
from realestate.metrics.index_metrics import MIN_HISTORY_MONTHS, region_metrics

st.set_page_config(page_title="하락 지역 순위", page_icon="📉", layout="wide")
st.title("하락 지역 순위")
st.write(
    "고점 대비 지수가 많이 내려온 지역 순서입니다. **많이 내려왔다고 싸다는 뜻은 아닙니다.** 최근 추세와 함께 보세요."
)

index, region, _ = common.load_rone()

# ── 필터: 한 줄에 ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns([2, 2, 1.3, 1.3])
types = c1.multiselect("지역 유형", REGION_TYPES, default=["시군구", "일반구"])
sidos = c2.multiselect("시도", sorted(region["sido"].unique()), placeholder="전체")
window = c3.selectbox(
    "고점을 찾을 기간",
    [60, 120, 36, None],
    format_func=lambda v: "전체 기간" if v is None else f"최근 {v // 12}년",
)
top_n = c4.slider("그래프에 보일 개수", 10, 40, 20, step=5)
hide_short = st.checkbox(
    f"통계 기간이 {MIN_HISTORY_MONTHS}개월보다 짧은 지역 빼기",
    value=True,
    help="행정구역 개편으로 새로 생긴 지역(인천 제물포구 등)은 통계가 몇 달뿐이라 고점 비교가 의미 없습니다.",
)

survey = index[index["source"] == "survey"]
m = region_metrics(survey, lookback_months=window).merge(region, on="region_key")
m = m[m["region_type"].isin(types)]
if sidos:
    m = m[m["sido"].isin(sidos)]
if hide_short:
    m = m[~m["short_history"]]
m = m.sort_values("drawdown_pct").reset_index(drop=True)

if m.empty:
    st.info("조건에 맞는 지역이 없습니다.")
    common.footer(disclaimer=True)
    st.stop()

st.caption(f"{common.fmt_ym(m['latest_ym'].max())} 기준 · {len(m)}개 지역 · 매매가격지수(조사)")

# ── 막대 그래프: 하락률 상위 ────────────────────────────────
top = m.head(top_n).iloc[::-1]  # 가장 많이 내려온 지역이 위로
color = common.series_colors()[0]
fig = go.Figure(
    go.Bar(
        x=top["drawdown_pct"],
        y=top["display_name"],
        orientation="h",
        marker={"color": color, "line": {"width": 0}},
        customdata=top[["peak_ym", "chg_3m_pct", "rebound_pct"]].assign(
            peak_ym=top["peak_ym"].dt.strftime("%Y-%m")
        ),
        hovertemplate=(
            "<b>%{y}</b><br>고점 대비 %{x:.1f}%<br>고점 %{customdata[0]}"
            "<br>최근 3개월 %{customdata[1]:+.2f}%<br>바닥 대비 반등 %{customdata[2]:.1f}%<extra></extra>"
        ),
        texttemplate="%{x:.1f}%",
        textposition="outside",
    )
)
fig.update_layout(
    height=max(320, 22 * len(top) + 60),
    margin={"l": 10, "r": 40, "t": 10, "b": 10},
    bargap=0.25,
    xaxis={"title": "고점 대비 (%)", "zeroline": True, "showgrid": True},
    yaxis={"showgrid": False},
)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")
st.caption(common.SOURCE_TEXT["rone"])

# ── 전체 표 ────────────────────────────────────────────────
st.subheader("전체 목록")
table = pd.DataFrame(
    {
        "순위": range(1, len(m) + 1),
        "지역": m["display_name"],
        "유형": m["region_type"],
        "고점 대비(%)": m["drawdown_pct"],
        "고점 월": m["peak_ym"].dt.strftime("%Y-%m"),
        "고점 후 개월": m["months_since_peak"],
        "바닥 대비 반등(%)": m["rebound_pct"],
        "3개월(%)": m["chg_3m_pct"],
        "12개월(%)": m["chg_12m_pct"],
        "현재 지수": m["latest"],
    }
)
st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    height=480,
    column_config={
        c: st.column_config.NumberColumn(format="%.2f")
        for c in ("고점 대비(%)", "바닥 대비 반등(%)", "3개월(%)", "12개월(%)", "현재 지수")
    },
)
st.download_button(
    "표를 CSV로 받기", table.to_csv(index=False).encode("utf-8-sig"), "하락_지역_순위.csv", "text/csv"
)

common.footer(disclaimer=True)
