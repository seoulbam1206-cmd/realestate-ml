"""한국부동산원 R-ONE 가로형 CSV → 세로형 표.

두 파일 모두 CP949, 지역이 행·월이 열인 가로형이고 헤더가 여러 줄이다 (DATA_CHECK.md).
- 매매가격지수(조사): 헤더 3줄, 지역 4칸, 월마다 [지수, 전월 대비 증감률] 2칸
- 실거래가격지수: 헤더 2줄, 지역 2칸, 월마다 [지수] 1칸
헤더 줄 수·지역 칸 수·월당 칸 수는 파일에서 직접 알아낸다.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

AGGREGATE_NAMES = {"전국", "수도권", "지방권", "지방", "5대광역시", "4대광역시", "8개도", "7개도"}
# 실거래가격지수 파일은 "지방", 조사 지수 파일은 "지방권"으로 표기한다
NAME_ALIASES = {"지방": "지방권"}

REGION_TYPES = ("묶음", "시도", "권역", "시군구", "일반구")

_MONTH_RE = re.compile(r"^\s*(\d{4})년\s*(\d{1,2})월\s*$")


@dataclass(frozen=True)
class WideCsv:
    header_rows: int
    region_cols: int
    values_per_month: int
    months: list[pd.Timestamp]
    rows: list[list[str]]


def parse_month(label: str) -> pd.Timestamp:
    m = _MONTH_RE.match(label)
    if not m:
        raise ValueError(f"월 형식이 아닙니다: {label!r}")
    return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)


def read_wide_csv(path: Path, encoding: str = "cp949") -> WideCsv:
    with open(path, encoding=encoding, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    header_rows = 0
    while header_rows < len(rows) and rows[header_rows][0].strip() == "No":
        header_rows += 1
    if header_rows == 0:
        raise ValueError(f"헤더(No, 지역, ...)를 찾지 못했습니다: {path.name}")

    first = rows[0]
    region_cols = sum(1 for c in first[1:] if c.strip() == "지역")
    labels = first[1 + region_cols :]
    unique = list(dict.fromkeys(labels))
    if len(labels) % len(unique) != 0:
        raise ValueError(f"월 열 개수가 맞지 않습니다: {path.name}")
    values_per_month = len(labels) // len(unique)
    for i, label in enumerate(unique):
        block = labels[i * values_per_month : (i + 1) * values_per_month]
        if any(b != label for b in block):
            raise ValueError(f"월 열 순서가 예상과 다릅니다: {label}")

    width = len(first)
    data = rows[header_rows:]
    bad = [r[0] for r in data if len(r) != width]
    if bad:
        raise ValueError(f"열 개수가 다른 행이 있습니다: No={bad[:5]}")
    return WideCsv(header_rows, region_cols, values_per_month, [parse_month(m) for m in unique], data)


def region_key(parts: list[str]) -> str:
    """'서울|강북지역|도심권|도심권' → '서울>강북지역>도심권' (같은 이름 반복 제거)."""
    out: list[str] = []
    for p in (NAME_ALIASES.get(p.strip(), p.strip()) for p in parts):
        if p and (not out or out[-1] != p):
            out.append(p)
    return ">".join(out)


def _to_float(text: str) -> float:
    text = text.strip().replace(",", "")
    return float(text) if text else float("nan")


def to_long(wide: WideCsv, source: str) -> pd.DataFrame:
    """세로형: source, region_key, ym, index_value, mom_pct."""
    n_months = len(wide.months)
    records = []
    for r in wide.rows:
        key = region_key(r[1 : 1 + wide.region_cols])
        values = r[1 + wide.region_cols :]
        for i in range(n_months):
            idx = _to_float(values[i * wide.values_per_month])
            if math.isnan(idx):  # 그 지역 통계가 아직 없던 달
                continue
            mom = (
                _to_float(values[i * wide.values_per_month + 1])
                if wide.values_per_month > 1
                else float("nan")
            )
            records.append((source, key, wide.months[i], idx, mom))
    df = pd.DataFrame(records, columns=["source", "region_key", "ym", "index_value", "mom_pct"])
    if df.duplicated(["region_key", "ym"]).any():
        raise ValueError(f"[{source}] 지역 키가 겹칩니다. region_key 규칙을 확인하세요.")

    df = df.sort_values(["region_key", "ym"]).reset_index(drop=True)
    first_obs = ~df.duplicated("region_key")
    if wide.values_per_month > 1:
        # 파일은 통계 시작 달의 증감률을 0.00 으로 적어 둔다 → 실제로는 알 수 없는 값
        df.loc[first_obs, "mom_pct"] = float("nan")
    else:
        df["mom_pct"] = (df.groupby("region_key")["index_value"].pct_change() * 100).round(2)
    return df


def classify(keys: list[str]) -> pd.DataFrame:
    """지역 계층표: region_key, name, sido, parent_key, depth, region_type, display_name."""
    rows = []
    types: dict[str, str] = {}
    for key in sorted(keys, key=lambda k: k.count(">")):
        parts = key.split(">")
        name = parts[-1]
        parent = ">".join(parts[:-1]) or None
        if len(parts) == 1:
            rtype = "묶음" if name in AGGREGATE_NAMES else "시도"
        elif parts[0] == "전남광주" and len(parts) == 2 and name in {"전남", "광주"}:
            rtype = "시도"  # 통합 전 시도. 시계열이 2003년부터 이어진다
        elif name.endswith(("권", "지역")):
            rtype = "권역"
        elif name.endswith("구") and types.get(parent or "") == "시군구" and parent.endswith("시"):
            rtype = "일반구"
        else:
            rtype = "시군구"
        types[key] = rtype
        rows.append((key, name, parts[0], parent, len(parts), rtype))
    df = pd.DataFrame(rows, columns=["region_key", "name", "sido", "parent_key", "depth", "region_type"])
    df["display_name"] = df.apply(_display_name, axis=1, types=types)
    return df


def _display_name(row: pd.Series, types: dict[str, str]) -> str:
    parts = row["region_key"].split(">")
    if len(parts) == 1:
        return parts[0]
    if parts[0] == "전남광주":
        # 통합 전 시도 이름으로 보여준다: '광주', '광주 동구', '전남 목포시'
        return " ".join(dict.fromkeys([parts[1], parts[-1]]))
    if row["region_type"] == "일반구":
        return f"{parts[0]} {parts[-2]} {parts[-1]}"
    return f"{parts[0]} {parts[-1]}"


def match_to_survey_keys(real_keys: list[str], survey_keys: list[str]) -> dict[str, str]:
    """실거래가격지수 지역 키를 조사 지수 지역 키에 맞춘다 (같은 시도, 같은 끝 이름)."""
    by_sido_name: dict[tuple[str, str], list[str]] = {}
    for k in survey_keys:
        parts = k.split(">")
        by_sido_name.setdefault((parts[0], parts[-1]), []).append(k)
    mapping = {}
    for k in real_keys:
        parts = k.split(">")
        cands = by_sido_name.get((parts[0], parts[-1]), [])
        if len(cands) == 1:
            mapping[k] = cands[0]
        elif k in survey_keys:
            mapping[k] = k
        else:
            raise ValueError(f"실거래가격지수 지역 {k!r} 를 조사 지수 지역과 맞추지 못했습니다: 후보 {cands}")
    return mapping


def build(survey_path: Path, real_path: Path, encoding: str = "cp949") -> tuple[pd.DataFrame, pd.DataFrame]:
    """두 CSV를 읽어 (rone_index, rone_region) 을 돌려준다."""
    survey = to_long(read_wide_csv(survey_path, encoding), "survey")
    real = to_long(read_wide_csv(real_path, encoding), "real")
    mapping = match_to_survey_keys(
        real["region_key"].unique().tolist(), survey["region_key"].unique().tolist()
    )
    real["region_key"] = real["region_key"].map(mapping)

    index = pd.concat([survey, real], ignore_index=True)
    region = classify(survey["region_key"].unique().tolist())
    region["has_real"] = region["region_key"].isin(set(real["region_key"]))
    span = index[index["source"] == "survey"].groupby("region_key")["ym"].agg(first_ym="min", last_ym="max")
    region = region.merge(span, left_on="region_key", right_index=True, how="left")
    order = {t: i for i, t in enumerate(REGION_TYPES)}
    region = region.sort_values(
        ["region_type", "sido", "region_key"], key=lambda s: s.map(order) if s.name == "region_type" else s
    )
    return index, region.reset_index(drop=True)
