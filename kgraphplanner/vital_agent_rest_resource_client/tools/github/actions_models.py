"""
Request/response models for github_actions_tool.

Mirrored from vital-agent-resource-rest at commit `cf9f411`
(vital_agent_resource_app/tools/github/actions_models.py). Field sets verified
against test_data/github/service_schemas.json by tests/test_github_schema_parity.py.

All eight operations. trigger_workflow and rerun_workflow are additionally gated
server-side by ALLOW_WORKFLOW_DISPATCH, which is off by default.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from kgraphplanner.vital_agent_rest_resource_client.tools.github.common_models import (
    GitHubRepoBase, GitHubOutputBase
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class GitHubActionsListWorkflowsInput(GitHubRepoBase):
    """List the workflows defined in a repository."""
    operation: Literal["list_workflows"] = Field(..., description="Operation to perform")
    max_results: Optional[int] = Field(30, description="Maximum workflows to return", ge=1, le=100)
    page: Optional[int] = Field(None, description="Page number for pagination", ge=1)


class GitHubActionsListRunsInput(GitHubRepoBase):
    """List workflow runs, most recent first."""
    operation: Literal["list_workflow_runs"] = Field(..., description="Operation to perform")
    workflow_id: Optional[str] = Field(None, description="Workflow id or filename to filter by")
    branch: Optional[str] = Field(None, description="Only runs on this branch")
    status: Optional[str] = Field(None, description="Filter by status or conclusion, e.g. failure")
    actor: Optional[str] = Field(None, description="Only runs triggered by this login")
    event: Optional[str] = Field(None, description="Only runs from this event, e.g. push")
    max_results: Optional[int] = Field(30, description="Maximum runs to return", ge=1, le=100)
    page: Optional[int] = Field(None, description="Page number for pagination", ge=1)


class GitHubActionsGetRunInput(GitHubRepoBase):
    """Get one workflow run."""
    operation: Literal["get_workflow_run"] = Field(..., description="Operation to perform")
    run_id: int = Field(..., description="Workflow run id", ge=1)


class GitHubActionsListJobsInput(GitHubRepoBase):
    """List the jobs and step results for a workflow run."""
    operation: Literal["list_run_jobs"] = Field(..., description="Operation to perform")
    run_id: int = Field(..., description="Workflow run id", ge=1)
    filter: Optional[Literal["latest", "all"]] = Field("latest", description="Which attempt's jobs")
    max_results: Optional[int] = Field(30, description="Maximum jobs to return", ge=1, le=100)
    page: Optional[int] = Field(
        None, description="Page to fetch; this operation reads exactly one page", ge=1
    )


class GitHubActionsRunLogsInput(GitHubRepoBase):
    """Fetch log text for a workflow run.

    GitHub returns logs as a zip archive; the service unpacks it and returns the
    tail of each job's log, since raw CI logs run to megabytes.
    """
    operation: Literal["get_run_logs"] = Field(..., description="Operation to perform")
    run_id: int = Field(..., description="Workflow run id", ge=1)
    max_lines_per_file: Optional[int] = Field(
        50, description="Tail lines kept per log file", ge=1, le=500
    )
    max_files: Optional[int] = Field(10, description="Maximum log files to include", ge=1, le=50)


# --- writes ----------------------------------------------------------------

class GitHubActionsTriggerInput(GitHubRepoBase):
    """Trigger a workflow_dispatch run."""
    operation: Literal["trigger_workflow"] = Field(..., description="Operation to perform")
    workflow_id: str = Field(..., description="Workflow id or filename", min_length=1)
    ref: str = Field(..., description="Branch or tag to run against", min_length=1)
    inputs: Optional[dict] = Field(None, description="Inputs declared by the workflow")


class GitHubActionsCancelRunInput(GitHubRepoBase):
    """Cancel an in-progress workflow run."""
    operation: Literal["cancel_workflow_run"] = Field(..., description="Operation to perform")
    run_id: int = Field(..., description="Workflow run id", ge=1)


class GitHubActionsRerunInput(GitHubRepoBase):
    """Re-run a completed workflow run."""
    operation: Literal["rerun_workflow"] = Field(..., description="Operation to perform")
    run_id: int = Field(..., description="Workflow run id", ge=1)
    failed_jobs_only: Optional[bool] = Field(False, description="Re-run only the failed jobs")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

class GitHubWorkflow(BaseModel):
    id: int = Field(..., description="Workflow id")
    name: str = Field(..., description="Workflow name")
    path: Optional[str] = Field(None, description="Path of the workflow file")
    state: Optional[str] = Field(None, description="active or disabled")
    html_url: Optional[str] = Field(None, description="Browser URL for the workflow")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class GitHubWorkflowRun(BaseModel):
    id: int = Field(..., description="Run id")
    name: Optional[str] = Field(None, description="Workflow name")
    status: Optional[str] = Field(None, description="queued, in_progress or completed")
    conclusion: Optional[str] = Field(None, description="success, failure, cancelled, skipped")
    branch: Optional[str] = Field(None, description="Branch the run was on")
    event: Optional[str] = Field(None, description="Event that triggered the run")
    actor: Optional[str] = Field(None, description="Login that triggered the run")
    head_sha: Optional[str] = Field(None, description="Commit sha")
    run_number: Optional[int] = Field(None, description="Sequential run number")
    run_attempt: Optional[int] = Field(None, description="Attempt number")
    html_url: Optional[str] = Field(None, description="Browser URL for the run")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class GitHubWorkflowStep(BaseModel):
    name: str = Field(..., description="Step name")
    status: Optional[str] = Field(None, description="Step status")
    conclusion: Optional[str] = Field(None, description="Step conclusion")
    number: Optional[int] = Field(None, description="Step number")


class GitHubWorkflowJob(BaseModel):
    id: int = Field(..., description="Job id")
    name: str = Field(..., description="Job name")
    status: Optional[str] = Field(None, description="Job status")
    conclusion: Optional[str] = Field(None, description="Job conclusion")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    html_url: Optional[str] = Field(None, description="Browser URL for the job")
    steps: List[GitHubWorkflowStep] = Field(default_factory=list, description="Step results")


class GitHubRunLog(BaseModel):
    filename: str = Field(..., description="Log file name within the run archive")
    lines: List[str] = Field(default_factory=list, description="Tail of the log file")
    truncated: bool = Field(False, description="True if earlier lines were dropped")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class GitHubActionsToolOutput(GitHubOutputBase):
    """Output model for github_actions_tool, shared by all of its operations."""
    tool: Literal["github_actions_tool"] = Field("github_actions_tool", description="Tool identifier")
    operation: str = Field(..., description="Operation that was performed")
    workflows: List[GitHubWorkflow] = Field(default_factory=list, description="Workflows from list operations")
    runs: List[GitHubWorkflowRun] = Field(default_factory=list, description="Runs from list operations")
    run: Optional[GitHubWorkflowRun] = Field(None, description="Run from single-run operations")
    jobs: List[GitHubWorkflowJob] = Field(default_factory=list, description="Jobs in a run")
    logs: List[GitHubRunLog] = Field(default_factory=list, description="Truncated log files")
    triggered: Optional[bool] = Field(None, description="True if a dispatch was accepted")
    dispatch_note: Optional[str] = Field(
        None,
        description="Explains that workflow_dispatch returns no run id, and how the run was located"
    )
    total_count: Optional[int] = Field(
        None, description="Corpus total reported by GitHub, where supplied."
    )


GITHUB_ACTIONS_OPERATION_MODELS = {
    "list_workflows": GitHubActionsListWorkflowsInput,
    "list_workflow_runs": GitHubActionsListRunsInput,
    "get_workflow_run": GitHubActionsGetRunInput,
    "list_run_jobs": GitHubActionsListJobsInput,
    "get_run_logs": GitHubActionsRunLogsInput,
    "trigger_workflow": GitHubActionsTriggerInput,
    "cancel_workflow_run": GitHubActionsCancelRunInput,
    "rerun_workflow": GitHubActionsRerunInput,
}
