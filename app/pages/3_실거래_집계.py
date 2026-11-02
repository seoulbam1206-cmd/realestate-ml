import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common

st.set_page_config(page_title="실거래 집계", page_icon="🧾", layout="wide")
st.title("실거래 집계")
common.intro(
    [
        "시군구를 골라 아파트 **매매 거래량**과 **중위 거래가**·**㎡당 중위가**가 달마다 어떻게 변했는지 비교합니다.",
        "국토교통부 실거래를 시군구 × 계약월로 모은 값이며, 개별 거래는 보여 주지 않습니다.",
        "최근 2개월은 신고가 덜 들어온 잠정치라 점선으로 따로 표시합니다.",
    ]
)

agg, backend = common.load_public("agg_region_month")
names, _ = common.load_public("region_name")
names["label"] = names["sido_nm"] + " " + names["sgg_nm"]
label = names.set_index("sgg_cd")["label"]
last_ym = agg["ym"].max()
prov_start = last_ym - pd.DateOffset(months=common.PROVISIONAL_MONTHS - 1)  # 이 달부터 잠정 (점선)
firm_ym = prov_start - pd.DateOffset(months=1)  # 마지막 확정 달
prov_label = "·".join(f"{d.year}.{d.month}" for d in pd.date_range(prov_start, last_ym, freq="MS"))

# ── 필터: 한 줄에 ───────────────────────────────────────────
c1, c2, c3 = st.columns([1.3, 3, 1.4])
sidos = sorted(names["sido_nm"].unique())
sido = c1.selectbox("시도", sidos, index=sidos.index("서울특별시") if "서울특별시" in sidos else 0)
in_sido = names[names["sido_nm"] == sido].sort_values("sgg_nm")
vol = agg[agg["ym"] >= last_ym - pd.DateOffset(months=11)].groupby("sgg_cd")["trade_cnt"].sum()
default = in_sido.assign(v=in_sido["sgg_cd"].map(vol).fillna(0)).nlargest(3, "v")["sgg_cd"].tolist()
picked = c2.multiselect(
    f"시군구 (최대 {common.MAX_SERIES}개)",
    in_sido["sgg_cd"].tolist(),
    default=default,
    format_func=lambda c: names.set_index("sgg_cd").at[c, "sgg_nm"],
    max_selections=common.MAX_SERIES,
    help="기본값: 최근 12개월 거래가 가장 많은 3곳",
)
years = sorted(agg["ym"].dt.year.unique())
start_year = c3.select_slider("시작 연도", options=years, value=years[0])

if not picked:
    st.info("시군구를 하나 이상 고르세요.")
    common.footer(sources=["molit_onbid"], contract_basis=True, invest_note=True)
    st.stop()

view = agg[agg["sgg_cd"].isin(picked) & (agg["ym"].dt.year >= start_year)].sort_values("ym")
short = names.set_index("sgg_cd")["sgg_nm"]

st.info(
    f"최근 {common.PROVISIONAL_MONTHS}개월({prov_label})은 {common.PROVISIONAL_NOTE}. "
    "그래프에서 점선과 회색 배경으로 구분하고, 아래 요약 숫자는 그 전 달 기준입니다.",
    icon="⏳",
)

# ── 요약 숫자: 첫 번째 시군구, 마지막 확정 달 ─────────────────
first = picked[0]
s = agg[(agg["sgg_cd"] == first) & (agg["ym"] <= firm_ym)].set_index("ym").sort_index()
cur = s.iloc[-1]
year_ago = s["median_price"].get(cur.name - pd.DateOffset(years=1))
st.markdown(
    f"**{label[first]}** · {common.fmt_ym(cur.name)} 계약분 (최근 {common.PROVISIONAL_MONTHS}개월 제외)"
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("거래량", f"{int(cur['trade_cnt']):,}건")
k2.metric(
    "중위 거래가",
    "-" if pd.isna(cur["median_price"]) else f"{cur['median_price'] / 1e4:.2f}억 원",
    None
    if pd.isna(cur["median_price"]) or pd.isna(year_ago)
    else f"1년 전 대비 {(cur['median_price'] / year_ago - 1) * 100:+.1f}%",
    delta_color="off",
)
k3.metric("㎡당 중위가", "-" if pd.isna(cur["median_ppm2"]) else f"{cur['median_ppm2']:,.0f}만 원")
k4.metric("해제 (지금 기준)", f"{int(cur['cancel_cnt']):,}건")

colors = common.series_colors()


def line_chart(col: str, y_title: str, fmt: str) -> go.Figure:
    fig = go.Figure()
    if view["ym"].max() >= prov_start:  # 잠정 구간 회색 배경
        fig.add_vrect(
            x0=prov_start - pd.Timedelta(days=15), x1=last_ym + pd.Timedelta(days=15),
            fillcolor="gray", opacity=0.12, line_width=0, layer="below",
            annotation_text="신고 진행 중", annotation_position="top left", annotation_font_size=11,
        )  # fmt: skip
    for i, c in enumerate(picked):
        d = view[view["sgg_cd"] == c]
        firm = d[d["ym"] < prov_start]
        prov = d[d["ym"] >= firm_ym]  # 확정 마지막 달에서 이어서 점선
        fig.add_trace(
            go.Scatter(
                x=firm["ym"], y=firm[col], name=short[c], mode="lines", connectgaps=False, legendgroup=c,
                line={"width": 2, "color": colors[i]},
                hovertemplate=f"{short[c]}: %{{y:{fmt}}}<extra></extra>",
            )
        )  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=prov["ym"], y=prov[col], mode="lines", legendgroup=c, showlegend=False, hoverinfo="skip",
                line={"width": 2, "color": colors[i], "dash": "dot"},
            )
        )  # fmt: skip
        p2 = d[d["ym"] >= prov_start]
        fig.add_trace(  # 잠정 달 hover (선과 겹치는 투명 점)
            go.Scatter(
                x=p2["ym"], y=p2[col], mode="markers", legendgroup=c, showlegend=False,
                marker={"size": 10, "color": colors[i], "opacity": 0},
                hovertemplate=f"{short[c]} (잠정): %{{y:{fmt}}}<extra></extra>",
            )
        )  # fmt: skip
        if len(picked) <= 4 and d[col].notna().any():
            last = d.dropna(subset=[col]).iloc[-1]
            fig.add_annotation(
                x=last["ym"], y=last[col], text=short[c], showarrow=False, xanchor="left", xshift=6
            )
    fig.update_layout(
        height=340,
        margin={"l": 10, "r": 90, "t": 10, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        yaxis_title=y_title,
        xaxis={"hoverformat": "%Y년 %m월", "showgrid": False},
        yaxis={"zeroline": False, "rangemode": "tozero" if col == "trade_cnt" else "normal"},
        showlegend=len(picked) > 1,
    )
    return fig


st.caption(f"{common.term('중위가')} · {common.term('해제')}")
st.subheader("거래량 (계약월)")
st.plotly_chart(line_chart("trade_cnt", "건", ",.0f"), width="stretch", theme="streamlit")
st.subheader("중위 거래가")
st.plotly_chart(line_chart("median_price", "만 원", ",.0f"), width="stretch", theme="streamlit")
st.subheader("㎡당 중위가")
st.plotly_chart(line_chart("median_ppm2", "만 원/㎡", ",.0f"), width="stretch", theme="streamlit")
st.caption(
    f"계약월 {common.fmt_ym(agg['ym'].min())} ~ {common.fmt_ym(last_ym)} · 지금 기준 해제 안 된 거래 · "
    f"거래가 5건 미만인 달은 가격을 비움(선이 끊김) · 점선·회색 배경 = 최근 {common.PROVISIONAL_MONTHS}개월({prov_label}), "
    f"{common.PROVISIONAL_NOTE} · 자료 위치: {backend}"
)

with st.expander("그래프 데이터 표로 보기"):
    t = view.assign(
        시군구=view["sgg_cd"].map(short),
        계약월=view["ym"].dt.strftime("%Y-%m") + view["ym"].ge(prov_start).map({True: " (잠정)", False: ""}),
    )
    st.dataframe(
        t[
            [
                "시군구",
                "계약월",
                "trade_cnt",
                "cancel_cnt",
                "median_price",
                "p25_ppm2",
                "median_ppm2",
                "p75_ppm2",
            ]
        ].rename(
            columns={
                "trade_cnt": "거래량",
                "cancel_cnt": "해제",
                "median_price": "중위 거래가(만 원)",
                "p25_ppm2": "㎡당 하위 25%",
                "median_ppm2": "㎡당 중위",
                "p75_ppm2": "㎡당 상위 25%",
            }
        ),  # fmt: skip
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="localized")
            for c in ("중위 거래가(만 원)", "㎡당 하위 25%", "㎡당 중위", "㎡당 상위 25%")
        },
    )

common.footer(sources=["molit_onbid"], contract_basis=True, invest_note=True)
