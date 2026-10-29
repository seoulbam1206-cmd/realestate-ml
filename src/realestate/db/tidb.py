"""TiDB Cloud 접속. 계정은 두 개로 나눈다 (PLAN.md 4-1).

- 쓰기 계정: 로컬 .env 의 TIDB_* → scripts/publish.py 만 사용
- 읽기 전용 계정: Streamlit Secrets [tidb] → 공개 앱만 사용. 접속 직후 권한을 검사해 쓰기 권한이 있으면 멈춘다
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote_plus

import certifi
from sqlalchemy import Engine, create_engine, text

SQL_DIR = Path(__file__).parent / "sql"

# 읽기 전용으로 인정하는 권한. 그 밖의 권한(INSERT, UPDATE, ALL PRIVILEGES 등)이 하나라도 있으면 거부
_READ_ONLY_PRIVS = {"SELECT", "USAGE", "SHOW VIEW"}
_GRANT_RE = re.compile(r"^GRANT\s+(.+?)\s+ON\s+", re.IGNORECASE)


def make_engine(host: str, port: int, user: str, password: str, database: str) -> Engine:
    url = f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}@{host}:{int(port)}/{database}"
    return create_engine(
        url,
        connect_args={"ssl": {"ca": certifi.where()}},  # TiDB Cloud 는 TLS 필수
        pool_pre_ping=True,
        pool_recycle=300,
    )


def write_privileges(grants: list[str]) -> set[str]:
    """SHOW GRANTS 결과에서 읽기 외 권한만 골라낸다. 비어 있으면 읽기 전용."""
    extra: set[str] = set()
    for g in grants:
        m = _GRANT_RE.match(g.strip())
        if not m:
            continue
        for priv in (p.strip().upper() for p in m.group(1).split(",")):
            if priv not in _READ_ONLY_PRIVS:
                extra.add(priv)
    return extra


def assert_read_only(engine: Engine) -> None:
    with engine.connect() as conn:
        grants = [row[0] for row in conn.execute(text("SHOW GRANTS"))]
    extra = write_privileges(grants)
    if extra:
        raise PermissionError(
            f"공개 앱에 읽기 전용이 아닌 계정이 연결됐습니다 (권한: {sorted(extra)}). "
            "Streamlit Secrets 에는 web_ro 계정만 넣으세요."
        )


def schema_sql() -> list[str]:
    body = (SQL_DIR / "schema.sql").read_text(encoding="utf-8")
    body = "\n".join(line for line in body.splitlines() if not line.strip().startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]
