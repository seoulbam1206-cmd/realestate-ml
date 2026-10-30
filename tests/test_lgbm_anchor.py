from datetime import date

from realestate.models.lgbm import anchor_for, valid_first_ym


def test_anchor_is_previous_quarter_end():
    assert anchor_for("202303") == date(2022, 12, 31)
    assert anchor_for("202307") == date(2023, 6, 30)
    assert anchor_for("202311") == date(2023, 9, 30)
    assert anchor_for("202401") == date(2023, 12, 31)
    assert anchor_for("202509") == date(2025, 6, 30)


def test_valid_months_are_last_fully_reported():
    # D − 30일 이 그 달 말일이면 그 달까지 완결, 아니면 전달까지
    assert valid_first_ym(date(2023, 9, 30)) == "202307"  # C = 8/31 → 7·8월
    assert valid_first_ym(date(2023, 12, 31)) == "202310"  # C = 12/1 → 10·11월
    assert valid_first_ym(date(2024, 3, 31)) == "202401"  # C = 3/1 → 1·2월
    assert valid_first_ym(date(2023, 6, 30)) == "202304"  # C = 5/31 → 4·5월
