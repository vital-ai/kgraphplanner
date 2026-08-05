"""
Case: GitHub model parity against the service schema snapshot.

The client-side models in vital_agent_rest_resource_client/tools/github/ are
hand-mirrored from the service. They drift, and the failure is silent -- a field
added upstream simply parses as None here. Upstream moved six times in one day
while these were being written, including breaking output-contract changes
(returned_count added, total_count narrowed, next_page added).

This compares our mirrors against a snapshot of the service's own OpenAPI schema
in test_data/github/service_schemas.json. Offline, so it runs anywhere.

To refresh the snapshot after a deliberate service change:

    curl -s http://localhost:8008/openapi.json > /tmp/o.json
    python test_scripts/cases/case_github_schema_parity.py --update /tmp/o.json

See planning/kg_tools/github_tools_plan.md section 7.
"""

from __future__ import annotations

import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

SNAPSHOT = os.path.join(project_root, "test_data", "github", "service_schemas.json")

from kgraphplanner.vital_agent_rest_resource_client.tools.github import issue_models

# Client model -> service schema name. Only the models we have mirrored so far;
# add entries as phases land rather than asserting over everything at once.
MIRRORED = {
    issue_models.GitHubIssueGetInput: "GitHubIssueGetInput",
    issue_models.GitHubIssueListInput: "GitHubIssueListInput",
    issue_models.GitHubIssueSearchInput: "GitHubIssueSearchInput",
    issue_models.GitHubIssue: "GitHubIssue",
    issue_models.GitHubComment: "GitHubComment",
    issue_models.GitHubIssueToolOutput: "GitHubIssueToolOutput",
}

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + ("" if condition else f" -- {detail}"))


def _service_fields(schema: dict, all_schemas: dict) -> set:
    """Field names for a schema, following allOf so inherited bases are included."""
    fields = set(schema.get("properties", {}))
    for part in schema.get("allOf", []):
        ref = part.get("$ref", "")
        if ref.startswith("#/components/schemas/"):
            fields |= _service_fields(all_schemas.get(ref.rsplit("/", 1)[-1], {}), all_schemas)
        else:
            fields |= set(part.get("properties", {}))
    return fields


def _service_required(schema: dict, all_schemas: dict) -> set:
    required = set(schema.get("required", []))
    for part in schema.get("allOf", []):
        ref = part.get("$ref", "")
        if ref.startswith("#/components/schemas/"):
            required |= _service_required(all_schemas.get(ref.rsplit("/", 1)[-1], {}), all_schemas)
        else:
            required |= set(part.get("required", []))
    return required


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--update":
        raw = json.load(open(sys.argv[2]))
        schemas = raw.get("components", {}).get("schemas", {})
        snap = {k: v for k, v in schemas.items() if k.startswith("GitHub")}
        existing = json.load(open(SNAPSHOT)) if os.path.exists(SNAPSHOT) else {}
        existing["schemas"] = snap
        json.dump(existing, open(SNAPSHOT, "w"), indent=2, sort_keys=True)
        print(f"snapshot updated: {len(snap)} schemas")
        return 0

    print("GitHub model parity against the service schema snapshot")
    print("=" * 62)
    snap = json.load(open(SNAPSHOT))
    schemas = snap["schemas"]
    print(f"snapshot: service commit {snap.get('_service_commit')}, "
          f"captured {snap.get('_captured')}, {len(schemas)} schemas\n")

    for model, service_name in MIRRORED.items():
        schema = schemas.get(service_name)
        if schema is None:
            check(f"{service_name}: present in the snapshot", False, "not found")
            continue

        ours = set(model.model_fields)
        theirs = _service_fields(schema, schemas)

        missing = theirs - ours
        check(f"{service_name}: no service field missing from our mirror",
              not missing, f"missing {sorted(missing)}")

        extra = ours - theirs
        check(f"{service_name}: no field we invented",
              not extra, f"we have {sorted(extra)} the service does not")

        # A field the service requires but we default is a silent way to send an
        # incomplete request; the reverse rejects requests the service accepts.
        ours_required = {n for n, f in model.model_fields.items() if f.is_required()}
        theirs_required = _service_required(schema, schemas) & ours
        req_diff = ours_required.symmetric_difference(theirs_required)
        check(f"{service_name}: required fields agree",
              not req_diff, f"differ on {sorted(req_diff)}")

    print("\n" + "=" * 62)
    print(f"Passed: {len(PASSED)}   Failed: {len(FAILED)}")
    for f in FAILED:
        print(f"  FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
