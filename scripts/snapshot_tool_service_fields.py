"""Record the tool service's result-model field names, for the parity test.

The tool service (vital-agent-resource-rest) is a separate repo, so
tests/test_tool_client_parity.py can't import it. This script reads the
service's models with ``ast`` (no service dependencies needed), resolves
inheritance within each tool package, and writes the field names of every
model the client mirrors to tests/data/tool_service_fields.json.

Run it after a service change, and commit the JSON together with the client
model changes:

    python scripts/snapshot_tool_service_fields.py ../vital-agent-resource-rest
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "data" / "tool_service_fields.json"

# Service tool package -> its result/output models the client mirrors.
# Keep in step with SERVICE_TO_CLIENT in tests/test_tool_client_parity.py.
MODELS = {
    "place_search": ["PlaceDetails", "PlaceSearchOutput"],
    # Not SerperRelatedSearch: the client keeps related_searches as plain dicts.
    "serper_web_search": ["SerperWebSearchResult", "SerperKnowledgeGraph",
                          "SerperPeopleAlsoAsk", "SerperWebSearchInput", "SerperWebSearchOutput"],
    "web_search": None,                 # None: every model in the package
    "weather": None,
    "google_address_validation": None,
    "github": None,
}


def package_classes(pkg: Path) -> dict[str, tuple[list[str], list[str]]]:
    """name -> (base names, own annotated fields) for every class in a package."""
    out = {}
    for f in sorted(pkg.glob("*.py")):
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, ast.ClassDef):
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                fields = [s.target.id for s in node.body
                          if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                          and not s.target.id.startswith("_")]
                out[node.name] = (bases, fields)
    return out


def resolved_fields(name: str, classes: dict) -> list[str]:
    bases, own = classes[name]
    fields: list[str] = []
    for base in bases:
        if base in classes:
            fields += [f for f in resolved_fields(base, classes) if f not in fields]
    return fields + [f for f in own if f not in fields]


def is_model(name: str, classes: dict) -> bool:
    bases = classes[name][0]
    return "BaseModel" in bases or any(b in classes and is_model(b, classes) for b in bases)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    service = Path(sys.argv[1]).resolve()
    tools = service / "vital_agent_resource_app" / "tools"
    snapshot = {"service_commit": subprocess.run(
        ["git", "-C", str(service), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip(), "models": {}}
    for pkg_name, wanted in MODELS.items():
        classes = package_classes(tools / pkg_name)
        names = wanted or sorted(n for n in classes if is_model(n, classes))
        for name in names:
            snapshot["models"][f"{pkg_name}.{name}"] = resolved_fields(name, classes)
    OUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(snapshot['models'])} models from service {snapshot['service_commit']} to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
