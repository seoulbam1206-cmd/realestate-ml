"""국토교통부 아파트 매매 실거래가 상세 API (RTMSDataSvcAptTradeDev).

- 인증키는 .env 의 DATA_GO_KR_KEY 에서만 읽는다. 인코딩된 키(%2B 등)면 디코딩해서 넘긴다 (httpx 가 다시 인코딩)
- 요청 URL에는 키가 들어가므로 URL·예외 메시지를 그대로 출력·저장하지 않는다 → mask() 를 거친다
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote

import httpx
import pandas as pd

from realestate.config import MOLIT_API_RAW, load_env, require_env, sources
from realestate.ingest import molit_csv

# 2026-10-29 첫 호출로 확인한 응답 항목 32개 (PLAN.md 6-1-2)
API_FIELDS = (
    "sggCd", "umdCd", "landCd", "bonbun", "bubun", "roadNm", "roadNmSggCd", "roadNmCd", "roadNmSeq",
    "roadNmbCd", "roadNmBonbun", "roadNmBubun", "umdNm", "aptNm", "jibun", "excluUseAr", "dealYear",
    "dealMonth", "dealDay", "dealAmount", "floor", "buildYear", "aptSeq", "cdealType", "cdealDay",
    "dealingGbn", "estateAgentSggNm", "rgstDate", "aptDong", "slerGbn", "buyerGbn", "landLeaseholdGbn",
)  # fmt: skip
OK_CODES = {"00", "000"}
NO_DATA_CODES = {"03"}  # 자료 없음 = 0건 성공
QUOTA_CODES = {"22"}  # 하루 호출 한도 초과 → 그날 수집 멈춤
FATAL_CODES = {"10", "11", "12", "20", "30", "31", "32"}  # 키·권한·요청 형식 문제 → 재시도해도 안 됨, 멈춤
# 그 밖의 코드(01, 02, 04, 05, 99), 응답이 XML이 아님, 네트워크 오류, HTTP 5xx → 재시도


@dataclass
class ApiResult:
    http_status: int | None
    result_code: str | None
    result_msg: str | None
    total_count: int | None
    items: list[dict[str, str]] = field(default_factory=list)
    raw: bytes = b""


def service_key() -> str:
    load_env()
    key = require_env("DATA_GO_KR_KEY").strip()
    return unquote(key) if "%" in key else key


def mask(text: str, key: str) -> str:
    for variant in {key, quote(key, safe=""), quote(key)}:
        if variant:
            text = text.replace(variant, "***")
    return text


def parse_xml(raw: bytes) -> tuple[str | None, str | None, int | None, list[dict[str, str]]]:
    root = ET.fromstring(raw)
    # 정상 응답: response/header/resultCode, 게이트웨이 오류: OpenAPI_ServiceResponse/cmmMsgHeader/returnReasonCode
    code = root.findtext(".//resultCode") or root.findtext(".//returnReasonCode")
    msg = root.findtext(".//resultMsg") or root.findtext(".//returnAuthMsg") or root.findtext(".//errMsg")
    total = root.findtext(".//totalCount")
    items = [{c.tag: (c.text or "").strip() for c in item} for item in root.iter("item")]
    return code, msg, int(total) if total else None, items


def fetch(
    lawd_cd: str, deal_ymd: str, page: int = 1, num_rows: int | None = None, timeout: float = 30
) -> ApiResult:
    cfg = sources()["molit_apt_trade"]
    key = service_key()
    url = f"{cfg['base_url']}/{cfg['operation']}"
    params = {
        "serviceKey": key,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ymd,
        "pageNo": page,
        "numOfRows": num_rows or cfg["num_of_rows"],
    }
    try:
        resp = httpx.get(url, params=params, timeout=timeout)
    except httpx.HTTPError as e:  # 메시지에 URL(키 포함)이 섞일 수 있음
        raise RuntimeError(f"{type(e).__name__}: {mask(str(e), key)}") from None
    raw = resp.content
    try:
        code, msg, total, items = parse_xml(raw)
    except ET.ParseError:
        return ApiResult(
            resp.status_code, None, mask(raw[:200].decode("utf-8", "replace"), key), None, [], raw
        )
    return ApiResult(resp.status_code, code, msg, total, items, raw)


def parse_cdeal_day(s: pd.Series) -> pd.Series:
    """해제사유발생일. CSV 는 YYYYMMDD, API 는 YY.MM.DD + cdealType="O" (2026-10-29 확인, 포항 남구 2026.1)."""
    s = s.fillna("").str.strip()
    out = pd.to_datetime(s.where(s.str.fullmatch(r"\d{8}")), format="%Y%m%d")
    short = pd.to_datetime(s.where(s.str.fullmatch(r"\d{2}\.\d{2}\.\d{2}")), format="%y.%m.%d")
    return out.fillna(short)


def to_clean(
    items: list[dict[str, str]], fetched_at: datetime, full_nm_by_bjd: dict[str, str]
) -> pd.DataFrame:
    """API item 목록 → CSV 정리본과 같은 칸 (+ API에만 있는 apt_seq 등). 키 계산도 CSV와 같다."""
    cols = sorted({k for it in items for k in it} | set(API_FIELDS))
    api = pd.DataFrame(items, columns=cols, dtype=object)  # 0건인 달도 문자열 칸
    api = api.fillna("").apply(lambda c: c.str.strip())
    na = lambda c: api[c].mask(api[c].isin(["", "-"]))  # noqa: E731
    bjd_cd = api["sggCd"] + api["umdCd"]
    df = pd.DataFrame(
        {
            "source": "api",
            "sgg_full_nm": bjd_cd.map(full_nm_by_bjd).astype(object).fillna(api["umdNm"]),
            "bjd_cd": bjd_cd,
            "sgg_cd": api["sggCd"],
            "umd_cd": api["umdCd"],
            "umd_nm": api["umdNm"],
            "jibun": na("jibun"),
            "bonbun": pd.to_numeric(api["bonbun"], errors="coerce").astype("Int32"),
            "bubun": pd.to_numeric(api["bubun"], errors="coerce").astype("Int32"),
            "apt_nm": api["aptNm"],
            "apt_dong": na("aptDong"),
            "exclu_use_ar": pd.to_numeric(api["excluUseAr"]),
            "floor": pd.to_numeric(api["floor"]).astype("Int16"),
            "build_year": pd.to_numeric(api["buildYear"], errors="coerce").astype("Int16"),
            "deal_date": pd.to_datetime(
                api["dealYear"] + api["dealMonth"].str.zfill(2) + api["dealDay"].str.zfill(2), format="%Y%m%d"
            ),
            "deal_amount": pd.to_numeric(api["dealAmount"].str.replace(",", "", regex=False)).astype("int64"),
            "dealing_gbn": na("dealingGbn"),
            "buyer_gbn": na("buyerGbn"),
            "seller_gbn": na("slerGbn"),
            "estate_agent_sgg_nm": na("estateAgentSggNm"),
            "road_nm": na("roadNm"),
            "cdeal_day": parse_cdeal_day(api["cdealDay"]),
            "rgst_date": pd.to_datetime(na("rgstDate"), format="%y.%m.%d"),  # 표시용. 피처 금지
            "cdeal_type": na("cdealType"),
            "apt_seq": na("aptSeq"),
            "land_leasehold_gbn": na("landLeaseholdGbn"),
        }
    )
    df["is_canceled"] = df["cdeal_day"].notna() | df["cdeal_type"].notna()
    df["price_per_m2"] = df["deal_amount"] / df["exclu_use_ar"]
    molit_csv.add_keys(df)
    df["first_seen_at"] = fetched_at
    df["last_seen_at"] = fetched_at
    return df


def save_raw(raw: bytes, lawd_cd: str, deal_ymd: str, page: int) -> Path:
    out = MOLIT_API_RAW / f"ym={deal_ymd}" / f"sgg={lawd_cd}_p{page}_{datetime.now():%Y%m%d%H%M%S}.xml.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(raw))
    return out
