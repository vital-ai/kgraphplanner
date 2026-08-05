"""GitHub client models vs the service's own OpenAPI schema.

The models in vital_agent_rest_resource_client/tools/github/ are hand-mirrored
from vital-agent-resource-rest. They drift, and the failure is silent: a field
added upstream simply parses as None here. On its first run this check caught
three divergences, all introduced locally by relaxing fields the service
requires -- which is exactly the kind of change that destroys drift detection.

The snapshot lives in test_data/github/service_schemas.json and is refreshed with
test_scripts/cases/case_github_schema_parity.py --update <openapi.json>.

Offline: reads the snapshot, never the service.
"""

import json
from pathlib import Path

import pytest

from kgraphplanner.vital_agent_rest_resource_client.tools.github import issue_models, pr_models, actions_models, code_models, repo_models

SNAPSHOT = Path(__file__).resolve().parent.parent / "test_data" / "github" / "service_schemas.json"

# Client model -> service schema name. Add an entry as each phase mirrors more
# models, so coverage is explicit rather than silently partial.
MIRRORED = {
    "GitHubIssueGetInput": issue_models.GitHubIssueGetInput,
    "GitHubIssueListInput": issue_models.GitHubIssueListInput,
    "GitHubIssueSearchInput": issue_models.GitHubIssueSearchInput,
    "GitHubIssue": issue_models.GitHubIssue,
    "GitHubComment": issue_models.GitHubComment,
    "GitHubIssueToolOutput": issue_models.GitHubIssueToolOutput,
    "GitHubIssueCommentListInput": issue_models.GitHubIssueCommentListInput,
    "GitHubPRListInput": pr_models.GitHubPRListInput,
    "GitHubPRGetInput": pr_models.GitHubPRGetInput,
    "GitHubPRFilesInput": pr_models.GitHubPRFilesInput,
    "GitHubPRCommentListInput": pr_models.GitHubPRCommentListInput,
    "GitHubPRReviewListInput": pr_models.GitHubPRReviewListInput,
    "GitHubPullRequest": pr_models.GitHubPullRequest,
    "GitHubPRFile": pr_models.GitHubPRFile,
    "GitHubPRComment": pr_models.GitHubPRComment,
    "GitHubPRReview": pr_models.GitHubPRReview,
    "GitHubPRToolOutput": pr_models.GitHubPRToolOutput,
    "GitHubActionsListWorkflowsInput": actions_models.GitHubActionsListWorkflowsInput,
    "GitHubActionsListRunsInput": actions_models.GitHubActionsListRunsInput,
    "GitHubActionsGetRunInput": actions_models.GitHubActionsGetRunInput,
    "GitHubActionsListJobsInput": actions_models.GitHubActionsListJobsInput,
    "GitHubActionsRunLogsInput": actions_models.GitHubActionsRunLogsInput,
    "GitHubWorkflow": actions_models.GitHubWorkflow,
    "GitHubWorkflowRun": actions_models.GitHubWorkflowRun,
    "GitHubWorkflowStep": actions_models.GitHubWorkflowStep,
    "GitHubWorkflowJob": actions_models.GitHubWorkflowJob,
    "GitHubRunLog": actions_models.GitHubRunLog,
    "GitHubActionsToolOutput": actions_models.GitHubActionsToolOutput,
    "GitHubIssueCreateInput": issue_models.GitHubIssueCreateInput,
    "GitHubIssueUpdateInput": issue_models.GitHubIssueUpdateInput,
    "GitHubIssueCloseInput": issue_models.GitHubIssueCloseInput,
    "GitHubIssueReopenInput": issue_models.GitHubIssueReopenInput,
    "GitHubIssueCommentCreateInput": issue_models.GitHubIssueCommentCreateInput,
    "GitHubIssueCommentUpdateInput": issue_models.GitHubIssueCommentUpdateInput,
    "GitHubIssueCommentDeleteInput": issue_models.GitHubIssueCommentDeleteInput,
    "GitHubIssueAddLabelsInput": issue_models.GitHubIssueAddLabelsInput,
    "GitHubIssueFindByBodyInput": issue_models.GitHubIssueFindByBodyInput,
    "GitHubIssueListLabelsInput": issue_models.GitHubIssueListLabelsInput,
    "GitHubIssueListMilestonesInput": issue_models.GitHubIssueListMilestonesInput,
    "GitHubIssueListAssignableUsersInput": issue_models.GitHubIssueListAssignableUsersInput,
    "GitHubLabel": issue_models.GitHubLabel,
    "GitHubMilestone": issue_models.GitHubMilestone,
    "GitHubIssueRemoveLabelsInput": issue_models.GitHubIssueRemoveLabelsInput,
    "GitHubIssueAddAssigneesInput": issue_models.GitHubIssueAddAssigneesInput,
    "GitHubIssueRemoveAssigneesInput": issue_models.GitHubIssueRemoveAssigneesInput,
    "GitHubPRCreateInput": pr_models.GitHubPRCreateInput,
    "GitHubPRUpdateInput": pr_models.GitHubPRUpdateInput,
    "GitHubPRCommentCreateInput": pr_models.GitHubPRCommentCreateInput,
    "GitHubPRReviewCreateInput": pr_models.GitHubPRReviewCreateInput,
    "GitHubPRRequestReviewersInput": pr_models.GitHubPRRequestReviewersInput,
    "GitHubActionsTriggerInput": actions_models.GitHubActionsTriggerInput,
    "GitHubActionsCancelRunInput": actions_models.GitHubActionsCancelRunInput,
    "GitHubActionsRerunInput": actions_models.GitHubActionsRerunInput,
    # github_code_tool and github_repo_tool -- added by the service's authority
    # split (commit 6543689) and the gap-closing work (cf9f411).
    "GitHubCreateBranchInput": code_models.GitHubCreateBranchInput,
    "GitHubDeleteBranchInput": code_models.GitHubDeleteBranchInput,
    "GitHubWriteFileInput": code_models.GitHubWriteFileInput,
    "GitHubDeleteFileInput": code_models.GitHubDeleteFileInput,
    "GitHubMergeInput": code_models.GitHubMergeInput,
    "GitHubWriteFilesInput": code_models.GitHubWriteFilesInput,
    "GitHubFileWrite": code_models.GitHubFileWrite,
    "GitHubCommitResult": code_models.GitHubCommitResult,
    "GitHubGetAuthenticatedUserInput": repo_models.GitHubGetAuthenticatedUserInput,
    "GitHubAuthenticatedUser": repo_models.GitHubAuthenticatedUser,
    "GitHubWriteResult": code_models.GitHubWriteResult,
    "GitHubMergeResult": code_models.GitHubMergeResult,
    "GitHubDeleteResult": code_models.GitHubDeleteResult,
    "GitHubCodeToolOutput": code_models.GitHubCodeToolOutput,
    "GitHubRepoGetInput": repo_models.GitHubRepoGetInput,
    "GitHubGetFileInput": repo_models.GitHubGetFileInput,
    "GitHubListBranchesInput": repo_models.GitHubListBranchesInput,
    "GitHubListCommitsInput": repo_models.GitHubListCommitsInput,
    "GitHubGetCommitInput": repo_models.GitHubGetCommitInput,
    "GitHubCompareRefsInput": repo_models.GitHubCompareRefsInput,
    "GitHubRepository": repo_models.GitHubRepository,
    "GitHubBranch": repo_models.GitHubBranch,
    "GitHubCommit": repo_models.GitHubCommit,
    "GitHubFileContent": repo_models.GitHubFileContent,
    "GitHubComparison": repo_models.GitHubComparison,
    "GitHubRepoToolOutput": repo_models.GitHubRepoToolOutput,
}


@pytest.fixture(scope="module")
def schemas():
    if not SNAPSHOT.exists():
        pytest.skip(f"no schema snapshot at {SNAPSHOT}")
    return json.loads(SNAPSHOT.read_text())["schemas"]


_BOUND_KEYS = ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems")


def _bounds(spec):
    """Value constraints on a field, looking through the anyOf that Optional
    produces -- Optional[int] with ge=1 puts the bound on the int branch, not
    at the top level."""
    branches = [spec] + [b for b in spec.get("anyOf", []) if b.get("type") != "null"]
    return {k: b[k] for b in branches for k in _BOUND_KEYS if k in b}


def _resolve(schema, all_schemas, key):
    """Collect `key` from a schema, following allOf so inherited bases count."""
    found = set(schema.get(key, {}) if key == "properties" else schema.get(key, []))
    for part in schema.get("allOf", []):
        ref = part.get("$ref", "")
        if ref.startswith("#/components/schemas/"):
            found |= _resolve(all_schemas.get(ref.rsplit("/", 1)[-1], {}), all_schemas, key)
        else:
            found |= set(part.get(key, {}) if key == "properties" else part.get(key, []))
    return found


@pytest.mark.parametrize("name", sorted(MIRRORED))
class TestParity:

    def test_schema_is_in_the_snapshot(self, name, schemas):
        assert name in schemas, f"{name} missing from the snapshot -- refresh it?"

    def test_no_service_field_is_missing_from_our_mirror(self, name, schemas):
        ours = set(MIRRORED[name].model_fields)
        theirs = _resolve(schemas[name], schemas, "properties")
        assert not (theirs - ours), (
            f"{name}: service has {sorted(theirs - ours)} and we do not -- "
            f"those values are being silently dropped"
        )

    def test_we_invented_no_fields(self, name, schemas):
        ours = set(MIRRORED[name].model_fields)
        theirs = _resolve(schemas[name], schemas, "properties")
        assert not (ours - theirs), (
            f"{name}: we have {sorted(ours - theirs)} the service does not"
        )

    def test_required_fields_agree(self, name, schemas):
        """Relaxing a required field is how drift detection gets lost: the mirror
        keeps parsing after the service changes, and nothing says so."""
        model = MIRRORED[name]
        ours = {n for n, f in model.model_fields.items() if f.is_required()}
        theirs = _resolve(schemas[name], schemas, "required") & set(model.model_fields)
        assert ours == theirs, (
            f"{name}: required differs -- ours {sorted(ours)}, service {sorted(theirs)}"
        )

    def test_value_constraints_agree(self, name, schemas):
        """Field names matching is not enough. A missing bound parses fine locally
        and is rejected by the service, so the mirror turns what should be a local
        validation error into a runtime 422 -- which is how max_chars was found.
        """
        model = MIRRORED[name]
        ours = model.model_json_schema().get("properties", {})
        theirs = schemas[name].get("properties", {})
        for field, their_spec in theirs.items():
            if field not in model.model_fields:
                continue  # covered by the missing-field test
            mismatch = _bounds(their_spec) != _bounds(ours.get(field, {}))
            assert not mismatch, (
                f"{name}.{field}: constraints differ -- ours {_bounds(ours.get(field, {}))}, "
                f"service {_bounds(their_spec)}"
            )
