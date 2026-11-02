import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common

st.set_page_config(page_title="백테스트 성적표", page_icon="📊", layout="wide")
st.title("백테스트 성적표")

ic, backend = common.load_public("bt_ic")
grade, _ = common.load_public("bt_grade")
base, _ = common.load_public("bt_baseline")
meta_df, _ = common.load_public("bt_meta")
meta = dict(zip(meta_df["k"], meta_df["v"], strict=True))
span = lambda k: f"{meta[k][:4]}.{int(meta[k][4:6])}~{meta[k][7:11]}.{int(meta[k][11:])}"  # noqa: E731
w = json.loads(meta["weights"])

st.write(
    f"점수 설정을 튜닝 기간({span('tuning_period')})에 고정한 뒤 **평가 기간({span('eval_period')}) 에 한 번만** 채점한 결과입니다. "
    f"정답은 그 달부터 6·12개월 동안 같은 단지·면적대 가격이 전국보다 얼마나 더 올랐는지(초과 수익)입니다. "
    f"확인 기간({span('check_period')})은 정답이 일부만 있어 참고용입니다."
)
st.warning(common.REGION_NOTE, icon="⚠️")
k1, k2, k3, k4 = st.columns(4)
k1.metric("판정 (미리 정한 3개 규칙)", "통과" if meta["verdict"] == "pass" else "미달")
k2.metric("평가 6개월 IC", meta["eval_ic6"], help="매월 시군구 순위 상관(점수 vs 이후 초과 수익)의 평균")
k3.metric("가중치 A·B·C·D", " · ".join(f"{w[k]:.2f}" for k in "ABCD"))
k4.metric(
    "규칙 버전", meta["rule_version"], help="2 = R1 을 절대 불가 대신 '데이터 부족'으로 (튜닝 후 조정 1회)"
)

f3 = lambda v: "-" if pd.isna(v) else f"{v:+.3f}"  # noqa: E731
pc = lambda v: "-" if pd.isna(v) else f"{v * 100:+.2f}"  # noqa: E731
PERIOD = {"tuning": "튜닝", "eval": "평가", "check": "확인 (참고)"}

# ── 1. IC ───────────────────────────────────────────────────
st.subheader("순위 상관 IC")
nat = ic[(ic["zone"] == "전국") & (ic["value_key"] == "score")]
rows = []
for p in PERIOD:
    for h in (6, 12):
        r = nat[(nat["period"] == p) & (nat["horizon_m"] == h)]
        if r.empty:
            continue
        r = r.iloc[0]
        ci = f"{f3(r['ci_lo'])} ~ {f3(r['ci_hi'])}" if pd.notna(r["ci_lo"]) else ""
        rows.append({"기간": PERIOD[p], "정답": f"{h}개월", "IC 평균": f3(r["ic_mean"]), "95% 구간": ci,
                     "IC 양수인 달": f"{r['pos_share'] * 100:.0f}%", "달 수": int(r["months"])})  # fmt: skip
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.markdown("**권역 안 IC (평가 기간, 사후 분석)** — 같은 권역 시군구끼리만 매월 순위 상관")
h_ic = st.radio("정답 기간", [6, 12], horizontal=True, format_func=lambda h: f"{h}개월", key="h_ic")
ev = ic[(ic["period"] == "eval") & (ic["horizon_m"] == h_ic)]
wide = []
for _key, g in ev.groupby("value_key", sort=False):
    row = {"값": g["value_name"].iloc[0]}
    for z in ("전국", "수도권", "지방"):
        r = g[g["zone"] == z].iloc[0]
        row[z] = (
            f"{f3(r['ic_mean'])}  [{f3(r['ci_lo'])} ~ {f3(r['ci_hi'])}]  양수 {r['pos_share'] * 100:.0f}%"
        )
    wide.append(row)
st.dataframe(pd.DataFrame(wide), hide_index=True, width="stretch")
st.caption(
    "[ ] 는 95% 구간 (월별 IC 원형 블록 부트스트랩 1,000회, 블록 = 정답 기간). 구간에 0 이 들어 있으면 우연과 구분되지 않음"
)

# ── 2. 등급별 성과 ─────────────────────────────────────────
st.subheader("등급별 이후 성과 (평가 기간)")
h = st.radio("정답 기간 ", [6, 12], horizontal=True, format_func=lambda h: f"{h}개월", key="h_grade")
g = grade[(grade["period"] == "eval") & (grade["horizon_m"] == h)]
ORDER = ["매매 추천", "매매 고려", "매매 고려 대상 아님", "매매 절대 불가", "데이터 부족"]
g = g.set_index("grade").reindex(ORDER).reset_index()

fig = go.Figure(
    go.Bar(
        x=g["grade"], y=g["x_mean"] * 100, marker={"color": common.series_colors()[0], "cornerradius": 4},
        text=[f"{v:+.2f}%" if pd.notna(v) else "정답 없음" for v in g["x_mean"] * 100], textposition="outside",
        customdata=g["n"], hovertemplate="%{x}<br>초과 수익 평균 %{y:+.2f}%<br>정답 있는 건 %{customdata:,}<extra></extra>",
    )
)  # fmt: skip
fig.update_layout(
    height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title=f"{h}개월 초과 수익 평균 (%)",
    yaxis={"zeroline": True, "zerolinewidth": 1}, xaxis={"showgrid": False}, showlegend=False,
)  # fmt: skip
st.plotly_chart(fig, width="stretch", theme="streamlit")

tbl = pd.DataFrame(
    {
        "등급": g["grade"],
        "시군구-월": g["n_all"],
        "정답 있음": g["n"],
        "초과 수익 평균 (%)": g["x_mean"].map(pc),
        "초과 수익 중앙 (%)": g["x_median"].map(pc),
        "적중률 (%)": g["hit"].map(lambda v: "-" if pd.isna(v) else f"{v * 100:.1f}"),
    }
)
if h == 12:
    tbl["12개월 최대 하락 평균 (%)"] = g["mdd_mean"].map(pc)
tbl["안내"] = (
    g["grade"]
    .map(
        {
            "매매 절대 불가": f"R2·R3 {common.UNVERIFIED} (정답 있는 건이 적음)",
            "데이터 부족": common.NODATA_NOTE,
        }
    )
    .fillna("")
)
st.dataframe(tbl, hide_index=True, width="stretch")
st.caption(
    "초과 수익 = 시군구 − 전국 (같은 단지·면적대 비교, log 차이 × 100). 적중률 = 초과 수익 > 0 인 비율. 최대 하락은 R-ONE 조사 지수"
)

with st.expander("튜닝 기간 · 확인 기간 (참고)"):
    for p in ("tuning", "check"):
        d = grade[(grade["period"] == p) & (grade["n"] > 0)]
        if d.empty:
            continue
        st.markdown(f"**{PERIOD[p]}**")
        st.dataframe(
            d.assign(
                정답=d["horizon_m"].astype(str) + "개월",
                평균=d["x_mean"].map(pc),
                적중률=d["hit"].map(lambda v: f"{v * 100:.1f}"),
            )[["정답", "grade", "n", "평균", "적중률"]].rename(
                columns={
                    "grade": "등급",
                    "n": "정답 있음",
                    "평균": "초과 수익 평균 (%)",
                    "적중률": "적중률 (%)",
                }
            ),
            hide_index=True,
            width="stretch",
        )

# ── 3. 기준선 ───────────────────────────────────────────────
st.subheader("기준선과 비교 (평가 기간, 추천 − 고려 대상 아님)")
b = base.assign(정답=base["horizon_m"].astype(str) + "개월")
b["차이 (%p)"] = [
    f"{pc(lo)} ~ {pc(hi)}" if "무작위" in n else pc(s)
    for n, s, lo, hi in zip(b["name"], b["spread"], b["lo"], b["hi"], strict=True)
]
b["95% 구간"] = [
    f"{pc(lo)} ~ {pc(hi)}" if pd.notna(lo) and "무작위" not in n else ""
    for n, lo, hi in zip(b["name"], b["lo"], b["hi"], strict=True)
]
st.dataframe(
    b[["정답", "name", "차이 (%p)", "95% 구간"]].rename(columns={"name": "점수"}),
    hide_index=True,
    width="stretch",
)
st.caption(
    '"A 만"·"B 만" = 그 요소 하나로 같은 등급 수만큼 자른 것. 무작위 = 같은 달 안에서 등급을 1,000번 섞었을 때 차이의 95% 범위'
)

# ── 4. 한계 ─────────────────────────────────────────────────
st.subheader("한계")
st.markdown(
    f"""
- **추천과 고려가 구분되지 않습니다.** 평가 6개월 초과 수익 추천 +0.27% · 고려 +0.26%, 12개월은 고려가 더 높음 (+0.69% · +0.41%)
- **권역 구성 효과**: 추천의 90%가 수도권이고 평가 기간 수도권이 더 올랐습니다. 같은 권역 안 IC 는 0 과 구분되지 않습니다 (6개월 수도권 −0.058 · 지방 −0.006)
- **A·B 단독 IC 는 음수**입니다. 많이 내린 곳·모델보다 싸게 거래된 곳이 이후 덜 올랐고, 양의 IC 는 C(추세)·D(거래량)에서 나왔습니다
- **절대 불가 규칙**: R2 {common.UNVERIFIED} (평가 기간 하락기 없음, 0건), R3 {common.UNVERIFIED} (평가 21건 중 정답 3건)
- 데이터 부족은 정답이 있는 비율이 6~8% 라 성과를 판단하지 못합니다
- 다음 버전은 2026.9 이후 데이터로 같은 설정을 다시 검증합니다. 위 사후 분석 결과로 이번 설정을 바꾸지는 않습니다
"""
)
st.caption(
    f"MLflow 평가 run `{meta['eval_run_id'][:8]}` · 설정 `{meta['weights_hash']}` · 만든 시각 {meta['built_at']} · 자료 위치: {backend}"
)

common.footer(sources=["molit_onbid"], disclaimer=True, invest_note=True)
