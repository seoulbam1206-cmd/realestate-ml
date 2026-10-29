from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from realestate.ingest import bjd, molit_api, molit_csv
from realestate.ingest import log as ingest_log

PREAMBLE = [
    "□ 본 서비스에서 제공하는 정보는 법적인 효력이 없으므로 참고용으로만 활용하시기 바랍니다.",
    "□ 신고정보가 실시간 변경, 해제되어 제공시점에 따라 공개건수 및 내용이 상이할 수 있는 점 참고하시기 바랍니다.",
    "□ 본 자료는 계약일 기준입니다.",
    "□ 통계자료 활용시에는 참고자료로만 활용하시기 바랍니다.",
    "",
    "* 문의 1533-2949",
    "□ 검색조건",
    "계약일자 : 2025-12-01 ~ 2025-12-31",
    "실거래구분 : 아파트(매매)",
    "주소구분 : 지번주소",
    "시도 : 전체",
    "시군구 : 전체",
    "읍면동 : 전체",
    "면적 : 전체",
    "금액선택 : 전체",
]
HEADER = ",".join(f'"{c}"' for c in molit_csv.EXPECTED_COLUMNS)
ROWS = [
    ["1", "서울특별시 종로구 창신동", "702", "0702", "0000", "창신쌍용1", "84.9200", "202512", "6", "69,500",
     "101", "7", "개인", "개인", "1992", "낙산길 10", "-", "중개거래", "서울 종로구", "26.02.27"],
    ["2", "서울특별시 종로구 창신동", "702", "0702", "0000", "창신쌍용1", "84.9200", "202512", "6", "69,500",
     "101", "7", "개인", "개인", "1992", "낙산길 10", "-", "중개거래", "서울 종로구", "-"],
    ["3", "경기도 부천시 원미구  중동", "1", "0001", "0002", "중동 마을", "59.0000", "202512", "31", "120,000",
     "-", "-1", "법인", "개인", "2001", "길주로 1", "20260105", "직거래", "-", "-"],
]  # fmt: skip


def _write_csv(
    tmp_path: Path, name="아파트(매매)_실거래가_20261029133640_2025-12.csv", preamble=PREAMBLE
) -> Path:
    lines = [f'"{p}"' for p in preamble] + [HEADER] + [",".join(f'"{v}"' for v in r) for r in ROWS]
    p = tmp_path / name
    p.write_bytes(("\n".join(lines) + "\n").encode("cp949"))
    return p


@pytest.fixture
def name_to_bjd():
    codes = pd.DataFrame(
        {
            "bjd_cd": ["1111017400", "4119210800", "4119510800"],
            "full_nm": ["서울특별시 종로구 창신동", "경기도 부천시 원미구 중동", "경기도 부천시 원미구 중동"],
            "status": ["존재", "존재", "폐지"],
        }
    )
    codes["is_active"] = codes["status"].eq("존재")
    return bjd.name_lookup(codes)


def test_name_lookup_prefers_active_code(name_to_bjd):
    assert name_to_bjd["경기도 부천시 원미구 중동"] == "4119210800"


def test_read_meta_finds_header_and_period(tmp_path):
    m = molit_csv.read_meta(_write_csv(tmp_path))
    assert m.header_line == 15
    assert (m.period_from, m.period_to) == (date(2025, 12, 1), date(2025, 12, 31))
    assert m.ym == "202512" and not m.name_mismatch
    assert m.downloaded_at.strftime("%Y%m%d%H%M%S") == "20261029133640"


def test_filename_month_mismatch_uses_inside_period(tmp_path):
    m = molit_csv.read_meta(_write_csv(tmp_path, name="아파트(매매)_실거래가_20261029134955_2026-01.csv"))
    assert m.ym == "202512" and m.name_month == "202601" and m.name_mismatch


def test_header_found_even_if_preamble_length_changes(tmp_path):
    m = molit_csv.read_meta(_write_csv(tmp_path, preamble=PREAMBLE[6:]))
    assert m.header_line == 9


def test_validate_rejects_non_nationwide(tmp_path):
    pre = [p.replace("시도 : 전체", "시도 : 서울특별시") for p in PREAMBLE]
    with pytest.raises(ValueError, match="전국 파일이 아님"):
        molit_csv.validate_meta(molit_csv.read_meta(_write_csv(tmp_path, preamble=pre)))


def test_parse_values(tmp_path, name_to_bjd):
    m = molit_csv.read_meta(_write_csv(tmp_path))
    df = molit_csv.parse(m, name_to_bjd)
    molit_csv.check_rows(m, df)
    assert df["deal_amount"].tolist() == [69500, 69500, 120000]  # 쉼표 제거
    assert df["is_canceled"].tolist() == [False, False, True]  # 해제사유발생일 있음
    assert df.loc[2, "cdeal_day"] == pd.Timestamp("2026-01-05")
    assert df.loc[0, "rgst_date"] == pd.Timestamp("2026-02-27") and pd.isna(df.loc[1, "rgst_date"])
    assert pd.isna(df.loc[2, "apt_dong"]) and df.loc[2, "floor"] == -1  # '-' → 빈값, 지하층
    assert df.loc[2, "sgg_full_nm"] == "경기도 부천시 원미구 중동"  # 공백 정리
    assert df["sgg_cd"].tolist() == ["11110", "11110", "41192"]
    # 내용이 같은 거래 2건은 순번으로 구분
    assert df["trade_key"].is_unique and df.loc[0, "trade_key"][:16] == df.loc[1, "trade_key"][:16]
    assert df.loc[0, "complex_key"] == df.loc[1, "complex_key"]


def test_pick_latest_per_month(tmp_path):
    a = molit_csv.read_meta(_write_csv(tmp_path, name="x_20261029100000_2025-12.csv"))
    b = molit_csv.read_meta(_write_csv(tmp_path, name="x_20261029110000_2025-12.csv"))
    use, superseded = molit_csv.pick_latest_per_month([a, b])
    assert use == [b] and superseded == [a]


def test_api_xml_parse_and_mask():
    raw = (
        b"<response><header><resultCode>000</resultCode><resultMsg>OK</resultMsg></header><body><items>"
        b"<item><sggCd>11110</sggCd><dealAmount>12,200</dealAmount><cdealType> </cdealType></item>"
        b"</items><totalCount>1</totalCount></body></response>"
    )
    code, msg, total, items = molit_api.parse_xml(raw)
    assert (code, msg, total) == ("000", "OK", 1)
    assert items == [{"sggCd": "11110", "dealAmount": "12,200", "cdealType": ""}]
    key = "ab+cd/e="  # 가짜 키 (비밀값 검사 패턴에 안 걸리게 짧게)
    masked = molit_api.mask(f"url?k=ab%2Bcd%2Fe%3D&x=1 {key}", key)
    assert "ab" not in masked and masked.count("***") == 2


def test_gateway_error_xml():
    raw = (
        b"<OpenAPI_ServiceResponse><cmmMsgHeader><errMsg>SERVICE ERROR</errMsg>"
        b"<returnReasonCode>22</returnReasonCode></cmmMsgHeader></OpenAPI_ServiceResponse>"
    )
    code, msg, _, items = molit_api.parse_xml(raw)
    assert code == "22" and msg == "SERVICE ERROR" and items == []


def test_ingest_log_and_quota(tmp_path):
    with ingest_log.connect(tmp_path / "log.duckdb") as con:
        ingest_log.write(
            con, run_id="r1", source="molit_csv", job_type="csv_import", target_key="ym=202512",
            status="success", started_at=pd.Timestamp.now().to_pydatetime(), rows_fetched=3, file_sha256="h1",
        )  # fmt: skip
        assert ingest_log.imported_hashes(con, "molit_csv") == {"h1"}
        assert ingest_log.add_api_calls(con, "molit_api", 1, 1000) == 1
        assert ingest_log.add_api_calls(con, "molit_api", 2, 1000) == 3
        with pytest.raises(ValueError):
            ingest_log.write(con, run_id="r2", source="x", job_type="x", target_key="x", status="weird",
                             started_at=pd.Timestamp.now().to_pydatetime())  # fmt: skip
