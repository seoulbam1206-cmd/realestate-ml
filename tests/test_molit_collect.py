from datetime import datetime

import pandas as pd
import pytest

from realestate.ingest import bjd, molit_api, molit_collect
from realestate.ingest import log as ingest_log
from realestate.ingest.molit_collect import Collector, FatalApiError, QuotaStop, RetryExhausted


def _item(day="5", amount="69,500", floor="7", **kw):
    it = {f: "" for f in molit_api.API_FIELDS}
    it.update(sggCd="11110", umdCd="17400", umdNm="창신동", aptNm="창신쌍용1", jibun="702", bonbun="0702",
              bubun="0000", excluUseAr="84.92", dealYear="2026", dealMonth="9", dealDay=day,
              dealAmount=amount, floor=floor, buildYear="1992", aptSeq="11110-90", dealingGbn="중개거래")  # fmt: skip
    it.update(kw)
    return it


def _res(code="000", items=(), total=None, http=200):
    items = list(items)
    return molit_api.ApiResult(http, code, "OK", len(items) if total is None else total, items, b"<x/>")


class FakeApi:
    """정해 둔 응답을 차례로 돌려준다. RuntimeError 를 넣으면 네트워크 오류처럼 던진다."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, lawd, ym, page, num_rows):
        self.calls.append(page)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def con(tmp_path):
    with ingest_log.connect(tmp_path / "log.duckdb") as c:
        yield c


def _collector(con, fake, cap=800, num_rows=1000):
    col = Collector(con, cap=cap, fetch=fake, sleep=lambda s: None, save_raw=None, max_retries=2)
    col.num_rows = num_rows
    return col


def test_pages_until_total(con):
    fake = FakeApi(_res(items=[_item("1"), _item("2")], total=5), _res(items=[_item("3"), _item("4")], total=5),
                   _res(items=[_item("5")], total=5))  # fmt: skip
    res = _collector(con, fake, num_rows=2).collect_month("11110", "202609")
    assert fake.calls == [1, 2, 3] and len(res.items) == 5 and res.api_calls == 3
    assert ingest_log.calls_today(con, "molit_api") == 3


def test_retry_then_success_counts_every_call(con):
    fake = FakeApi(RuntimeError("ConnectTimeout"), _res(code="99"), _res(items=[_item()]))
    res = _collector(con, fake).collect_month("11110", "202609")
    assert res.retries == 2 and res.api_calls == 3 and len(res.items) == 1


def test_retry_exhausted(con):
    fake = FakeApi(_res(code="01"), _res(code="01"), _res(code="01"))
    with pytest.raises(RetryExhausted) as e:
        _collector(con, fake).collect_month("11110", "202609")
    assert e.value.partial.api_calls == 3


def test_cap_stops_before_calling(con):
    ingest_log.add_api_calls(con, "molit_api", 800, 1000)
    fake = FakeApi()
    with pytest.raises(QuotaStop):
        _collector(con, fake, cap=800).collect_month("11110", "202609")
    assert fake.calls == [] and ingest_log.calls_today(con, "molit_api") == 800


def test_cap_reached_mid_month(con):
    ingest_log.add_api_calls(con, "molit_api", 799, 1000)
    fake = FakeApi(_res(items=[_item("1"), _item("2")], total=4))
    with pytest.raises(QuotaStop) as e:
        _collector(con, fake, cap=800, num_rows=2).collect_month("11110", "202609")
    assert fake.calls == [1] and e.value.partial.api_calls == 1


def test_server_quota_and_fatal_codes_do_not_retry(con):
    with pytest.raises(QuotaStop):
        _collector(con, FakeApi(_res(code="22"))).collect_month("11110", "202609")
    with pytest.raises(QuotaStop):
        _collector(con, FakeApi(_res(code=None, http=429))).collect_month("11110", "202609")
    with pytest.raises(FatalApiError):
        _collector(con, FakeApi(_res(code="30"))).collect_month("11110", "202609")
    assert ingest_log.calls_today(con, "molit_api") == 3


def test_no_data_is_success(con):
    res = _collector(con, FakeApi(_res(code="03", total=0))).collect_month("11110", "202609")
    assert res.items == [] and res.pages == 1


def test_to_clean_matches_csv_keys_and_cancel():
    items = [_item(), _item(), _item("6", cdealType="O", cdealDay="26.10.02")]
    df = molit_api.to_clean(items, datetime(2026, 10, 29), {"1111017400": "서울특별시 종로구 창신동"})
    assert df["trade_key"].is_unique and df["deal_amount"].tolist() == [69500, 69500, 69500]
    assert df["is_canceled"].tolist() == [False, False, True]
    assert df.loc[2, "cdeal_day"] == pd.Timestamp("2026-10-02")
    assert df.loc[0, "sgg_full_nm"] == "서울특별시 종로구 창신동" and df.loc[0, "bonbun"] == 702
    # CSV 정리본과 같은 칸 이름 (API에만 있는 칸은 추가)
    assert {"trade_key", "complex_key", "deal_date", "exclu_use_ar", "is_canceled"} <= set(df.columns)


def test_cdeal_day_formats():
    s = pd.Series(["20261002", "26.10.02", "", None])
    out = molit_api.parse_cdeal_day(s)
    assert out[0] == out[1] == pd.Timestamp("2026-10-02") and out[2:].isna().all()


def test_save_clean_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(molit_collect, "APT_TRADE_CLEAN", tmp_path)
    t0, t1 = datetime(2026, 10, 1), datetime(2026, 10, 29)
    first = molit_api.to_clean([_item("1"), _item("2")], t0, {})
    assert molit_collect.save_clean(first, "11110", "202609") == {
        "rows_new": 2,
        "rows_removed": 0,
        "rows_canceled": 0,
    }
    second = molit_api.to_clean([_item("1", cdealType="O", cdealDay="26.10.20"), _item("3")], t1, {})
    counts = molit_collect.save_clean(second, "11110", "202609")
    assert counts == {"rows_new": 1, "rows_removed": 1, "rows_canceled": 1}
    saved = pd.read_parquet(molit_collect.clean_path("11110", "202609"))
    assert saved.loc[saved["deal_date"].dt.day == 1, "first_seen_at"].iloc[0] == pd.Timestamp(t0)


def test_lawd_codes_drop_parent_city_only():
    codes = pd.DataFrame({"bjd_cd": ["4111000000", "4111100000", "4374000000", "4374500000", "4100000000"],
                          "full_nm": ["경기도 수원시", "경기도 수원시 장안구", "충청북도 영동군", "충청북도 증평군", "경기도"],
                          "status": ["존재"] * 5})  # fmt: skip
    codes["sgg_cd"], codes["is_active"] = codes["bjd_cd"].str[:5], True
    assert bjd.lawd_codes(codes) == ["41111", "43740", "43745"]


def test_twin_trades_get_same_key_in_any_order():
    twins = [_item("5"), _item("5", cdealType="O", cdealDay="26.10.02")]
    a = molit_api.to_clean(twins, datetime(2026, 10, 29), {})
    b = molit_api.to_clean(twins[::-1], datetime(2026, 10, 29), {})
    assert dict(zip(a["trade_key"], a["is_canceled"], strict=True)) == dict(
        zip(b["trade_key"], b["is_canceled"], strict=True)
    )


def test_to_clean_zero_items():
    df = molit_api.to_clean([], datetime(2026, 10, 29), {})
    assert df.empty and "trade_key" in df.columns
