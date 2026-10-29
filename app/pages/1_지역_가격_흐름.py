import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
from realestate.metrics.index_metrics import rebase, region_metrics

st.set_page_config(page_title="지역 가격 흐름", page_icon="📈", layout="wide")
st.title("지역 가격 흐름")

index, region, _ = common.load_rone()
by_name = region.set_index("display_name")
options = region["display_name"].tolist()

# ── 필터: 한 줄에 ───────────────────────────────────────────
c1, c2, c3 = st.columns([3, 1.4, 1.6])
picked = c1.multiselect(
    "지역 (최대 6개)",
    options,
    default=["전국", "서울", "지방권"],
    max_selections=common.MAX_SERIES,
    placeholder="지역 이름을 입력하세요",
)
source = c2.radio(
    "지수",
    ["survey", "real"],
    format_func=common.INDEX_LABEL.get,
    horizontal=True,
    help="실거래가격지수는 시도·서울 권역 등 29개 지역만 있습니다.",
)
years = sorted(index["ym"].dt.year.unique())
start_year = c3.select_slider("시작 연도", options=years, value=max(years[0], 2015))

if not picked:
    st.info("지역을 하나 이상 고르세요.")
    common.footer()
    st.stop()

keys = by_name.loc[picked, "region_key"].tolist()
data = index[(index["source"] == source) & index["region_key"].isin(keys)]
missing = [n for n, k in zip(picked, keys, strict=True) if k not in set(data["region_key"])]
if missing:
    st.warning(f"{common.INDEX_LABEL[source]}에 없는 지역은 빠졌습니다: {', '.join(missing)}")

rebase_on = st.toggle(
    "시작 연도 1월 = 100 으로 다시 맞추기",
    help="지역끼리 같은 출발점에서 얼마나 오르내렸는지 비교할 때 씁니다.",
)
view = data[data["ym"].dt.year >= start_year]
if rebase_on and not view.empty:
    view = rebase(view, view["ym"].min())

# ── 요약 숫자: 첫 번째 지역 ────────────────────────────────
if not data.empty:
    first_key = next(k for k in keys if k in set(data["region_key"]))
    first_name = picked[keys.index(first_key)]
    m5 = region_metrics(data[data["region_key"] == first_key], lookback_months=60).iloc[0]
    st.markdown(f"**{first_name}** · {common.fmt_ym(m5['latest_ym'])} 기준")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("현재 지수", f"{m5['latest']:.2f}")
    k2.metric(
        "최근 5년 고점 대비", f"{m5['drawdown_pct']:+.1f}%", help=f"고점: {common.fmt_ym(m5['peak_ym'])}"
    )
    k3.metric("최근 3개월", f"{m5['chg_3m_pct']:+.2f}%")
    k4.metric("최근 12개월", f"{m5['chg_12m_pct']:+.2f}%")

# ── 선 그래프 ───────────────────────────────────────────────
colors = common.series_colors()
fig = go.Figure()
for i, (name, key) in enumerate(zip(picked, keys, strict=True)):
    s = view[view["region_key"] == key].sort_values("ym")
    if s.empty:
        continue
    fig.add_trace(
        go.Scatter(
            x=s["ym"],
            y=s["index_value"],
            name=name,
            mode="lines",
            line={"width": 2, "color": colors[i]},
            hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
        )
    )
    if len(picked) <= 4:  # 4개 이하일 때만 끝에 이름 직접 표시
        fig.add_annotation(
            x=s["ym"].iloc[-1],
            y=s["index_value"].iloc[-1],
            text=name,
            showarrow=False,
            xanchor="left",
            xshift=6,
            font={"size": 12},
        )
fig.update_layout(
    height=440,
    margin={"l": 10, "r": 80, "t": 10, "b": 10},
    hovermode="x unified",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    yaxis_title="지수 (시작 연도 1월=100)" if rebase_on else "지수 (2026년 6월=100)",
    xaxis={"hoverformat": "%Y년 %m월", "showgrid": False},
    yaxis={"gridwidth": 1, "zeroline": False},
)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")
st.caption(f"{common.INDEX_LABEL[source]} · {common.SOURCE_TEXT['rone']}")

# ── 지역별 지표 표 ─────────────────────────────────────────
st.subheader("고점 대비 지표")
window = st.radio(
    "고점을 찾을 기간",
    [60, 120, None],
    horizontal=True,
    format_func=lambda v: "전체 기간" if v is None else f"최근 {v // 12}년",
)
metrics = region_metrics(data, lookback_months=window)
if not metrics.empty:
    names = dict(zip(keys, picked, strict=True))
    table = pd.DataFrame(
        {
            "지역": metrics["region_key"].map(names),
            "현재 지수": metrics["latest"],
            "고점": metrics["peak"],
            "고점 월": metrics["peak_ym"].dt.strftime("%Y-%m"),
            "고점 대비(%)": metrics["drawdown_pct"],
            "고점 이후 바닥 대비 반등(%)": metrics["rebound_pct"],
            "3개월(%)": metrics["chg_3m_pct"],
            "6개월(%)": metrics["chg_6m_pct"],
            "12개월(%)": metrics["chg_12m_pct"],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.2f")
            for c in table.columns
            if c not in ("지역", "고점 월")
        },
    )

with st.expander("그래프 데이터 표로 보기"):
    wide = view.assign(지역=view["region_key"].map(dict(zip(keys, picked, strict=True)))).pivot(
        index="ym", columns="지역", values="index_value"
    )
    wide.index = wide.index.strftime("%Y-%m")
    st.dataframe(wide.round(2), use_container_width=True)

common.footer()
