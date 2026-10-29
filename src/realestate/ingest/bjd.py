"""법정동 코드 전체자료(code.go.kr) → 코드표, 이름 → 코드 매칭.

파일: CP949, 탭 구분, `법정동코드 / 법정동명 / 폐지여부` 3칸 (생성일·폐지일 없음, PLAN.md 5-2).
이름 끝 공백("경기도 부천시 원미구 ")이 있어 공백을 정리해서 비교한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_name(name: str) -> str:
    return " ".join(str(name).split())


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="cp949", dtype=str)
    df.columns = ["bjd_cd", "bjd_nm", "status"]
    df["full_nm"] = df["bjd_nm"].map(normalize_name)
    df["sgg_cd"] = df["bjd_cd"].str[:5]
    df["umd_cd"] = df["bjd_cd"].str[5:]
    df["is_active"] = df["status"].eq("존재")
    return df[["bjd_cd", "full_nm", "sgg_cd", "umd_cd", "status", "is_active"]]


def name_lookup(codes: pd.DataFrame) -> dict[str, str]:
    """전체 이름 → 법정동 코드. 같은 이름이 여러 개면 '존재' 코드를, 그래도 여러 개면 그 이름은 뺀다."""
    out: dict[str, str] = {}
    for name, g in codes.groupby("full_nm"):
        active = g[g["is_active"]]
        pick = active if len(active) else g
        if len(pick) == 1:
            out[name] = pick["bjd_cd"].iloc[0]
    return out


def lawd_codes(codes: pd.DataFrame) -> list[str]:
    """API 조회용 시군구 코드(LAWD_CD) 목록: 지금 존재하는 시군구, 일반구가 있는 시는 구 코드만.

    예: 수원시(41110)는 빼고 장안구(41111)·권선구(41113)... 를 쓴다. 증평군(43745)처럼 코드만 비슷한
    군은 시가 아니므로 이름("충청북도 영동군 " 으로 시작하는 하위 구가 있는지)으로 판단한다.
    """
    sgg = codes[
        codes["is_active"] & codes["bjd_cd"].str.endswith("00000") & ~codes["sgg_cd"].str.endswith("000")
    ]
    names = dict(zip(sgg["sgg_cd"], sgg["full_nm"], strict=True))
    parents = {c for c, n in names.items() if any(o.startswith(n + " ") for o in names.values())}
    return sorted(set(names) - parents)
