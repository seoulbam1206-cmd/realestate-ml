"""complex_map: CSV 식 단지키(complex_key) ↔ API 단지 일련번호(apt_seq) 대조표 (PLAN.md 5-2, 6-1-2).

CSV 에는 단지 번호가 없어 complex_key = 법정동 + 본번 + 부번 + 단지명(공백 제거) 해시로 단지를 구분한다.
API 행에는 둘 다 있으므로 API 행에서 짝을 모으고, API 에 한 번도 안 나온 CSV 단지는 지번으로 찾는다.

match_method
- api_row    : API 행에 complex_key 와 apt_seq 가 같이 있음 (가장 확실). 한 complex_key 에 apt_seq 가 여럿이면
               가장 많이 나온 것, conflict=True
- jibun      : API 에 없던 CSV 단지. 같은 법정동·본번·부번에 API 단지(apt_seq)가 하나뿐 → 그것
- jibun_name : 같은 지번에 API 단지가 여럿 → 단지명(공백 제거)이 서로 포함되는 것이 하나뿐이면 그것
- (없음)     : 못 찾음. API 로 다시 받은 달이 늘면 (매일 이어받기 점검) 다시 실행해서 채움

결과: data/clean/reference/complex_map.parquet, reports/complex_map.md (건수 요약만)
실행: uv run python scripts/build_complex_map.py
"""

from __future__ import annotations

import sys
from datetime import datetime

import duckdb
import pandas as pd

from realestate.config import APT_TRADE_CLEAN, BJD_RAW, REFERENCE_CLEAN, ROOT
from realestate.ingest import bjd, trades
from realestate.ingest.molit_csv import normalize_apt_name

OUT = REFERENCE_CLEAN / "complex_map.parquet"
REPORTS = ROOT / "reports"
COMPLEX_COLS = ["complex_key", "bjd_cd", "sgg_cd", "bonbun", "bubun", "apt_nm"]


def load_complexes(con: duckdb.DuckDBPyConnection, files: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ", ".join(f"'{p.as_posix()}'" for p in files)
    src = f"read_parquet([{paths}], union_by_name = true)"
    # API 짝: 이번에 API 로 받은 행만 (CSV 에서 넘어온 행은 apt_seq 가 없음)
    pairs = con.execute(
        f"""
        SELECT complex_key, apt_seq, any_value(bjd_cd) AS bjd_cd, any_value(bonbun) AS bonbun,
               any_value(bubun) AS bubun, any_value(apt_nm) AS apt_nm, count(*) AS n
        FROM {src} WHERE source = 'api' AND apt_seq IS NOT NULL GROUP BY 1, 2
        """
    ).df()
    # 모든 단지 (CSV·API, 고른 출처 기준) + 거래 수·마지막 거래월
    complexes = con.execute(
        f"""
        SELECT complex_key, any_value(bjd_cd) AS bjd_cd, any_value(sgg_cd) AS sgg_cd,
               any_value(bonbun) AS bonbun, any_value(bubun) AS bubun, mode(apt_nm) AS apt_nm,
               count(*) AS n_trades, max(strftime(deal_date, '%Y%m')) AS last_ym
        FROM {src} GROUP BY 1
        """
    ).df()
    return pairs, complexes


def build(pairs: pd.DataFrame, complexes: pd.DataFrame) -> pd.DataFrame:
    # 1) api_row
    first = pairs.sort_values(["complex_key", "n"], ascending=[True, False]).drop_duplicates("complex_key")
    n_seq = pairs.groupby("complex_key")["apt_seq"].nunique()
    direct = first.assign(match_method="api_row", conflict=first["complex_key"].map(n_seq) > 1)
    direct = direct[["complex_key", "apt_seq", "match_method", "conflict"]]

    # 2) API 단지 목록 (지번 → apt_seq). apt_seq 하나에 이름이 여럿일 수 있어 이름 목록으로
    seqs = (
        pairs.groupby(["bjd_cd", "bonbun", "bubun", "apt_seq"])["apt_nm"]
        .agg(lambda s: sorted(set(normalize_apt_name(s))))
        .reset_index()
    )
    rest = complexes[~complexes["complex_key"].isin(direct["complex_key"])]
    cand = rest[COMPLEX_COLS].merge(seqs, on=["bjd_cd", "bonbun", "bubun"], suffixes=("", "_api"))
    n_cand = cand.groupby("complex_key")["apt_seq"].transform("nunique")
    one = cand[n_cand == 1].drop_duplicates("complex_key").assign(match_method="jibun")

    many = cand[n_cand > 1].copy()
    nm = normalize_apt_name(many["apt_nm"])
    many["hit"] = [
        any(a and b and (a in b or b in a) for b in names)
        for a, names in zip(nm, many["apt_nm_api"], strict=True)
    ]
    hits = many[many["hit"]]
    by_name = hits[hits.groupby("complex_key")["apt_seq"].transform("nunique") == 1].drop_duplicates(
        "complex_key"
    )
    by_name = by_name.assign(match_method="jibun_name")
    found = pd.concat([one, by_name])[["complex_key", "apt_seq", "match_method"]].assign(conflict=False)

    out = complexes.merge(pd.concat([direct, found]), on="complex_key", how="left")
    out["n_jibun_candidates"] = out["complex_key"].map(cand.groupby("complex_key")["apt_seq"].nunique())
    out["built_at"] = datetime.now()
    return out


def write_report(cmap: pd.DataFrame, n_api_months: int, n_months: int) -> None:
    by = cmap["match_method"].fillna("(없음)").value_counts()
    trades_by = cmap.groupby(cmap["match_method"].fillna("(없음)"))["n_trades"].sum()
    total_c, total_t = len(cmap), cmap["n_trades"].sum()
    recent = cmap[cmap["last_ym"] >= "202510"]
    seq_multi = cmap.dropna(subset=["apt_seq"]).groupby("apt_seq")["complex_key"].nunique()
    unmatched = cmap[cmap["apt_seq"].isna()]
    lines = [
        "# complex_map — CSV 단지키 ↔ API 단지번호",
        "",
        f"- 만든 시각: {datetime.now():%Y-%m-%d %H:%M} · 스크립트 `scripts/build_complex_map.py`",
        f"- 범위: 정리본 {n_months}개월 중 API 로 읽는 달 {n_api_months}개월 (나머지는 CSV)",
        "",
        "| 방법 | 단지 | 단지 비율 | 거래 | 거래 비율 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in ["api_row", "jibun", "jibun_name", "(없음)"]:
        c, t = by.get(m, 0), trades_by.get(m, 0)
        lines.append(f"| {m} | {c:,} | {c / total_c:.1%} | {t:,} | {t / total_t:.1%} |")
    lines += [
        f"| **합계** | **{total_c:,}** | | **{total_t:,}** | |",
        "",
        f"- 최근 12개월(2025.10~) 거래가 있는 단지: {len(recent):,}개 중 매칭 {recent['apt_seq'].notna().mean():.1%}",
        f"- 한 complex_key 에 apt_seq 가 여럿(conflict): {int(cmap['conflict'].fillna(False).sum()):,}개 → 가장 많이 나온 것 사용",
        f"- 한 apt_seq 에 complex_key 가 여럿 (단지명 표기가 바뀐 경우 등): {int((seq_multi > 1).sum()):,}개 apt_seq",
        f"- 못 찾은 단지 중 같은 지번에 후보가 여럿이라 못 고른 것: {int(unmatched['n_jibun_candidates'].notna().sum()):,}개, "
        f"지번에 API 단지가 아예 없는 것: {int(unmatched['n_jibun_candidates'].isna().sum()):,}개",
        "",
        "## 해석",
        "",
        "- 못 찾은 단지는 대부분 2026.9 이전에만 거래가 있어 API 에 아직 안 나온 단지. 매일 이어받기의 점검이",
        "  2025.10~2026.8 을 API 로 다시 받으면(약 10일) 최근 12개월 단지는 대부분 api_row 로 채워짐 → 그때 다시 실행",
        "- 2025.9 이전에만 거래된 단지는 API 로 받지 않는 한 지번 매칭이 최선. MVP 3 모델은 complex_key 를 기본 단지 ID로 쓰고",
        "  apt_seq 는 온비드·호가 입력 연결(MVP 4)에만 사용",
        "",
    ]
    (REPORTS / "complex_map.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    lawds = bjd.lawd_codes(bjd.load(BJD_RAW))
    src = trades.month_sources(lawds, APT_TRADE_CLEAN)
    files = [p for f in src["files"] for p in f]
    con = duckdb.connect()
    pairs, complexes = load_complexes(con, files)
    cmap = build(pairs, complexes)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmap.to_parquet(OUT, index=False)
    REPORTS.mkdir(exist_ok=True)
    write_report(cmap, int((src["use"] == "api").sum()), len(src))
    by = cmap["match_method"].fillna("(없음)").value_counts().to_dict()
    print(f"단지 {len(cmap):,}개: {by} → {OUT.relative_to(ROOT)}, reports/complex_map.md")


if __name__ == "__main__":
    main()
