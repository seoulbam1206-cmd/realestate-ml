"""국토교통부 실거래가 공개시스템 CSV(아파트 매매) → 정리본 표.

파일 모양 (서울 샘플·전국 월별 파일로 확인, PLAN.md 6-1-2 · 6-1-3)
- CP949, 맨 위 설명(공지 + 검색조건)이 있고 그 뒤 `"NO","시군구",...` 헤더, 20개 컬럼
- 검색조건의 `계약일자 : 2025-12-01 ~ 2025-12-31` 이 이 파일의 실제 기간
- 파일 이름 끝의 `_YYYY-MM` 은 사람이 붙인 표시라 틀릴 수 있다 → 파일 안 기간을 기준으로 쓰고, 다르면 경고
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "NO", "시군구", "번지", "본번", "부번", "단지명", "전용면적(㎡)", "계약년월", "계약일", "거래금액(만원)",
    "동", "층", "매수자", "매도자", "건축년도", "도로명", "해제사유발생일", "거래유형", "중개사소재지", "등기일자",
]  # fmt: skip
MAX_PREAMBLE_LINES = 40
_PERIOD_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})")
_NAME_MONTH_RE = re.compile(r"_(\d{4})-(\d{2})\.csv$", re.IGNORECASE)
_DOWNLOAD_TS_RE = re.compile(r"_(\d{14})")


@dataclass(frozen=True)
class CsvMeta:
    path: Path
    header_line: int  # 0부터 센 헤더 줄 번호 (= 건너뛸 설명 줄 수)
    conditions: dict[str, str]
    period_from: date
    period_to: date
    name_month: str | None  # 파일 이름의 YYYYMM
    downloaded_at: datetime | None

    @property
    def ym(self) -> str:
        return f"{self.period_from:%Y%m}"

    @property
    def name_mismatch(self) -> bool:
        return self.name_month is not None and self.name_month != self.ym


def read_meta(path: Path, encoding: str = "cp949") -> CsvMeta:
    conditions: dict[str, str] = {}
    header_line = None
    with open(path, encoding=encoding) as f:
        for i, line in enumerate(f):
            text = line.strip().strip('"')
            if text.startswith('NO","시군구') or line.startswith('"NO","시군구"'):
                header_line = i
                break
            if " : " in text:
                k, v = text.split(" : ", 1)
                conditions[k.strip()] = v.strip()
            if i >= MAX_PREAMBLE_LINES:
                break
    if header_line is None:
        raise ValueError(f'{path.name}: 헤더 줄("NO","시군구")을 찾지 못했습니다')
    m = _PERIOD_RE.search(conditions.get("계약일자", ""))
    if not m:
        raise ValueError(f"{path.name}: 검색조건에 계약일자 기간이 없습니다")
    period_from, period_to = (date.fromisoformat(x) for x in m.groups())

    nm = _NAME_MONTH_RE.search(path.name)
    ts = _DOWNLOAD_TS_RE.search(path.name)
    return CsvMeta(
        path=path,
        header_line=header_line,
        conditions=conditions,
        period_from=period_from,
        period_to=period_to,
        name_month=f"{nm.group(1)}{nm.group(2)}" if nm else None,
        downloaded_at=datetime.strptime(ts.group(1), "%Y%m%d%H%M%S") if ts else None,
    )


def validate_meta(meta: CsvMeta) -> None:
    """한 파일 = 전국 · 아파트(매매) · 지번주소 · 한 달 전체 여야 한다."""
    c = meta.conditions
    problems = []
    if c.get("실거래구분") != "아파트(매매)":
        problems.append(f"실거래구분={c.get('실거래구분')}")
    if c.get("주소구분") not in (None, "지번주소"):
        problems.append(f"주소구분={c.get('주소구분')}")
    if c.get("시도") != "전체":
        problems.append(f"시도={c.get('시도')} (전국 파일이 아님)")
    month_end = (pd.Timestamp(meta.period_from) + pd.offsets.MonthEnd(0)).date()
    if meta.period_from.day != 1 or meta.period_to != month_end:
        problems.append(f"기간 {meta.period_from}~{meta.period_to} 이 한 달 전체가 아님")
    if problems:
        raise ValueError(f"{meta.path.name}: " + ", ".join(problems))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dash_to_na(s: pd.Series) -> pd.Series:
    s = s.str.strip()
    return s.mask(s.isin(["", "-"]))


def _key(parts: pd.DataFrame) -> pd.Series:
    joined = parts.astype(str).agg("|".join, axis=1)
    return joined.map(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest()[:16])


def normalize_apt_name(s: pd.Series) -> pd.Series:
    return s.fillna("").str.replace(r"\s+", "", regex=True)


def parse(meta: CsvMeta, name_to_bjd: dict[str, str], encoding: str = "cp949") -> pd.DataFrame:
    raw = pd.read_csv(
        meta.path, encoding=encoding, skiprows=meta.header_line, dtype=str, keep_default_na=False
    )
    if list(raw.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{meta.path.name}: 컬럼이 예상과 다릅니다: {list(raw.columns)}")

    full_nm = raw["시군구"].map(lambda x: " ".join(x.split()))
    bjd_cd = full_nm.map(name_to_bjd)
    cdeal = _dash_to_na(raw["해제사유발생일"])
    rgst = _dash_to_na(raw["등기일자"])
    deal_date = pd.to_datetime(raw["계약년월"] + raw["계약일"].str.zfill(2), format="%Y%m%d")

    df = pd.DataFrame(
        {
            "source": "csv",
            "ym": raw["계약년월"],
            "sgg_full_nm": full_nm,
            "bjd_cd": bjd_cd,
            "sgg_cd": bjd_cd.str[:5],
            "umd_cd": bjd_cd.str[5:],
            "umd_nm": full_nm.str.split(" ").str[-1],
            "jibun": _dash_to_na(raw["번지"]),
            "bonbun": pd.to_numeric(raw["본번"], errors="coerce").astype("Int32"),
            "bubun": pd.to_numeric(raw["부번"], errors="coerce").astype("Int32"),
            "apt_nm": raw["단지명"].str.strip(),
            "apt_dong": _dash_to_na(raw["동"]),
            "exclu_use_ar": pd.to_numeric(raw["전용면적(㎡)"]),
            "floor": pd.to_numeric(raw["층"]).astype("Int16"),
            "build_year": pd.to_numeric(raw["건축년도"], errors="coerce").astype("Int16"),
            "deal_date": deal_date,
            "deal_amount": pd.to_numeric(raw["거래금액(만원)"].str.replace(",", "", regex=False)).astype(
                "int64"
            ),
            "dealing_gbn": _dash_to_na(raw["거래유형"]),
            "buyer_gbn": _dash_to_na(raw["매수자"]),
            "seller_gbn": _dash_to_na(raw["매도자"]),
            "estate_agent_sgg_nm": _dash_to_na(raw["중개사소재지"]),
            "road_nm": _dash_to_na(raw["도로명"]),
            "cdeal_day": pd.to_datetime(cdeal, format="%Y%m%d"),
            "rgst_date": pd.to_datetime(rgst, format="%y.%m.%d"),  # 표시용. 계약 뒤 정보라 피처 금지
        }
    )
    df["is_canceled"] = df["cdeal_day"].notna()
    df["price_per_m2"] = df["deal_amount"] / df["exclu_use_ar"]  # 만원/㎡

    loc = df["bjd_cd"].fillna("NM:" + df["sgg_full_nm"])
    apt = normalize_apt_name(df["apt_nm"])
    df["complex_key"] = _key(pd.DataFrame({"loc": loc, "bon": df["bonbun"], "bu": df["bubun"], "apt": apt}))
    base = _key(
        pd.DataFrame(
            {
                "loc": loc, "bon": df["bonbun"], "bu": df["bubun"], "apt": apt,
                "ar": df["exclu_use_ar"].round(4), "fl": df["floor"], "dong": df["apt_dong"].fillna(""),
                "dd": df["deal_date"].dt.strftime("%Y%m%d"), "amt": df["deal_amount"],
            }
        )
    )  # fmt: skip
    df["trade_key"] = base + "-" + df.groupby(base).cumcount().astype(str)
    df["first_seen_at"] = meta.downloaded_at
    df["source_file"] = meta.path.name
    return df


def check_rows(meta: CsvMeta, df: pd.DataFrame) -> None:
    outside = df[(df["deal_date"].dt.date < meta.period_from) | (df["deal_date"].dt.date > meta.period_to)]
    if len(outside):
        raise ValueError(f"{meta.path.name}: 기간 밖 계약일 {len(outside)}건")
    if df["trade_key"].duplicated().any():
        raise ValueError(f"{meta.path.name}: trade_key 중복")


def pick_latest_per_month(metas: list[CsvMeta]) -> tuple[list[CsvMeta], list[CsvMeta]]:
    """같은 달 파일이 여러 개면 가장 늦게 받은 것만 쓴다. (쓸 것, 밀려난 것)"""
    by_month: dict[str, list[CsvMeta]] = {}
    for m in metas:
        by_month.setdefault(m.ym, []).append(m)
    use, superseded = [], []
    for ym in sorted(by_month):
        group = sorted(by_month[ym], key=lambda m: (m.downloaded_at or datetime.min, m.path.name))
        use.append(group[-1])
        superseded += group[:-1]
    return use, superseded
