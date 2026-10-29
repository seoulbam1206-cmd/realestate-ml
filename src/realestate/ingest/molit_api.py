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

from realestate.config import MOLIT_API_RAW, load_env, require_env, sources

OK_CODES = {"00", "000"}


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


def save_raw(raw: bytes, lawd_cd: str, deal_ymd: str, page: int) -> Path:
    out = MOLIT_API_RAW / f"ym={deal_ymd}" / f"sgg={lawd_cd}_p{page}_{datetime.now():%Y%m%d%H%M%S}.xml.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(raw))
    return out
