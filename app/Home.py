import streamlit as st

st.set_page_config(page_title="부동산 가격 흐름", page_icon="🏠", layout="wide")

import common  # noqa: E402
from realestate.metrics.index_metrics import region_metrics  # noqa: E402

index, region, backend = common.load_rone()
survey = index[index["source"] == "survey"]
latest_ym = survey["ym"].max()

st.title("부동산 가격 흐름")
st.write("전국 아파트 가격이 어디서 얼마나 내려왔는지 추적하는 데이터 프로젝트입니다.")
common.banner()

st.subheader(f"지금 상황 · {common.fmt_ym(latest_ym)} 기준")
names = region.set_index("display_name")["region_key"]
headline = ["전국", "수도권", "서울", "지방권"]
m = region_metrics(survey[survey["region_key"].isin(names[headline])], lookback_months=60).set_index(
    "region_key"
)
cols = st.columns(len(headline))
for col, name in zip(cols, headline, strict=True):
    r = m.loc[names[name]]
    col.metric(
        name,
        f"{r['drawdown_pct']:+.1f}%",
        f"최근 3개월 {r['chg_3m_pct']:+.2f}%",
        delta_color="normal",
        help=f"최근 5년 고점({common.fmt_ym(r['peak_ym'])}) 대비 현재 매매가격지수. 아래 작은 글씨는 최근 3개월 변화율.",
    )
st.caption("큰 숫자: 최근 5년 고점 대비 현재 지수 (0%면 지금이 고점)")

st.subheader("페이지 안내")
st.page_link(
    "pages/1_지역_가격_흐름.py",
    label="지역 가격 흐름 — 지역을 골라 지수 추이와 고점 대비 하락률 비교",
    icon="📈",
)
st.page_link(
    "pages/2_하락_지역_순위.py", label="하락 지역 순위 — 고점 대비 많이 내려온 시군구 순위", icon="📉"
)
st.page_link("pages/9_출처_안내.py", label="출처 · 안내 — 데이터 출처, 지수 설명, 면책", icon="📄")

with st.expander("데이터 정보"):
    n_regions = survey["region_key"].nunique()
    st.write(
        f"- 매매가격지수(조사): {n_regions}개 지역, {common.fmt_ym(survey['ym'].min())} ~ {common.fmt_ym(latest_ym)}, 월간\n"
        f"- 실거래가격지수: {index.loc[index['source'] == 'real', 'region_key'].nunique()}개 지역, "
        f"{common.fmt_ym(index.loc[index['source'] == 'real', 'ym'].min())} ~ "
        f"{common.fmt_ym(index.loc[index['source'] == 'real', 'ym'].max())}\n"
        "- 두 지수 모두 2026년 6월 = 100\n"
        f"- 데이터 위치: {backend}"
    )

common.footer()
