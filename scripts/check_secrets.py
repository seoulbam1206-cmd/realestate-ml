"""커밋 전 비밀값 검사. 값은 절대 출력하지 않고, 걸린 파일 이름만 알려준다.

실행:
  uv run python scripts/check_secrets.py            # 커밋 대기(staged) 파일 검사 (pre-commit 훅이 부름)
  uv run python scripts/check_secrets.py --all      # git 이 관리하는 모든 파일 + 커밋 기록 전체 검사 (배포 전)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATHS = (
    re.compile(r"(^|/)\.env$"),
    re.compile(r"secrets\.toml$"),
    re.compile(r"^data/(raw|clean)/"),
)
PATTERNS = (
    re.compile(r"serviceKey=[A-Za-z0-9%+/=]{20,}"),
    re.compile(r"(?i)(password|passwd|secret|api_key)\s*[=:]\s*['\"][^'\"\s]{8,}['\"]"),
)


def env_values() -> list[str]:
    path = ROOT / ".env"
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            v = line.split("=", 1)[1].strip().strip("'\"")
            if len(v) >= 8:
                # 공공데이터포털 키는 인코딩/디코딩 두 형태가 있어 둘 다 찾는다
                values += list({v, quote(v, safe=""), unquote(v)})
    return values


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def scan_text(text: str, secrets: list[str]) -> bool:
    return any(s in text for s in secrets) or any(p.search(text) for p in PATTERNS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    secrets = env_values()
    problems: list[str] = []

    if args.all:
        files = git("ls-files").splitlines()
        history = git("log", "-p", "--all")
        if scan_text(history, secrets):
            problems.append("커밋 기록(git log -p) 안에 비밀값이 있습니다")
        read = lambda f: (ROOT / f).read_bytes()  # noqa: E731
    else:
        files = git("diff", "--cached", "--name-only", "--diff-filter=ACM").splitlines()
        read = lambda f: git("show", f":{f}").encode("utf-8", "replace")  # noqa: E731

    for f in files:
        if any(p.search(f) for p in FORBIDDEN_PATHS):
            problems.append(f"{f}: 커밋하면 안 되는 파일")
            continue
        try:
            text = read(f).decode("utf-8", "replace")
        except (OSError, subprocess.CalledProcessError):
            continue
        if scan_text(text, secrets):
            problems.append(f"{f}: 비밀값 또는 비밀값 같은 문자열")

    if problems:
        print("비밀값 검사 실패 (값은 표시하지 않음):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"비밀값 검사 통과: 파일 {len(files)}개" + (" + 커밋 기록" if args.all else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
