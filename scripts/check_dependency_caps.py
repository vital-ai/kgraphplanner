"""Report whether kgraphplanner's dependency caps hold back current releases.

Reads the upper bounds (``<X``) from pyproject.toml, asks PyPI for each
package's latest release, and prints whether the cap excludes it. For the
authlib cap it also checks the condition for lifting it: weaviate-client's
latest release must accept httpx 1.x or later (see the pyproject comment).

    python scripts/check_dependency_caps.py           # report only
    python scripts/check_dependency_caps.py --strict  # exit 1 if a cap holds back a release

Needs network access to pypi.org.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def pypi(package: str, version: str | None = None) -> dict:
    url = f"https://pypi.org/pypi/{package}/{version + '/' if version else ''}json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def capped_requirements() -> list[Requirement]:
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    reqs = [Requirement(d) for d in deps]
    return [r for r in reqs if any(s.operator in ("<", "<=") for s in r.specifier)]


def latest_release(package: str) -> Version:
    releases = pypi(package)["releases"]
    return max(Version(v) for v, files in releases.items()
               if files and not Version(v).is_prerelease)


def weaviate_accepts_httpx_1() -> tuple[bool, str]:
    """The authlib cap's lift condition: weaviate-client moved off httpx 0.x."""
    version = str(latest_release("weaviate-client"))
    requires = pypi("weaviate-client", version)["info"].get("requires_dist") or []
    httpx = [Requirement(r) for r in requires if Requirement(r).name in ("httpx", "httpx2")]
    accepts = any(r.name == "httpx2" or r.specifier.contains("1.0.0") for r in httpx)
    return accepts, f"weaviate-client {version}: {', '.join(str(r) for r in httpx) or 'no httpx requirement'}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any cap excludes the latest release")
    args = parser.parse_args()

    held_back = []
    for req in capped_requirements():
        latest = latest_release(req.name)
        allowed = req.specifier.contains(latest)
        print(f"{req.name:<18} cap {str(req.specifier):<16} latest {str(latest):<10} "
              f"{'ok' if allowed else 'HELD BACK'}")
        if not allowed:
            held_back.append(req.name)

    if "authlib" in held_back:
        accepts, detail = weaviate_accepts_httpx_1()
        print(f"\nauthlib lift condition ({detail}): "
              f"{'met, try lifting the cap' if accepts else 'not met, keep the cap'}")

    return 1 if args.strict and held_back else 0


if __name__ == "__main__":
    sys.exit(main())
