from realestate.db.tidb import schema_sql, write_privileges


def test_read_only_grants_pass():
    grants = [
        "GRANT USAGE ON *.* TO 'abc.web_ro'@'%'",
        "GRANT SELECT ON `realestate_public`.* TO 'abc.web_ro'@'%'",
    ]
    assert write_privileges(grants) == set()


def test_writer_grants_are_caught():
    grants = [
        "GRANT USAGE ON *.* TO 'abc.writer'@'%'",
        "GRANT SELECT,INSERT,UPDATE,DELETE ON `realestate_public`.* TO 'abc.writer'@'%'",
    ]
    assert write_privileges(grants) == {"INSERT", "UPDATE", "DELETE"}


def test_root_is_caught():
    assert write_privileges(["GRANT ALL PRIVILEGES ON *.* TO 'abc.root'@'%' WITH GRANT OPTION"]) == {
        "ALL PRIVILEGES"
    }


def test_schema_has_mvp1_tables():
    stmts = " ".join(schema_sql())
    for table in ("rone_region", "rone_index", "publish_meta"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in stmts


def test_publish_refuses_trade_level_columns():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import publish

    assert {"trade_key", "deal_amount", "deal_date"} <= publish.FORBIDDEN_COLS
    assert set(publish.GROUPS["mvp3"]) == {
        "region_name", "agg_region_month", "region_score", "bt_ic", "bt_grade", "bt_baseline", "bt_meta"
    }  # fmt: skip
    assert set(publish.GROUPS["complex"]) == {"complex_price"}
    ddl = " ".join(schema_sql())
    assert all(f"CREATE TABLE IF NOT EXISTS {t} " in ddl for t in publish.TABLES)
