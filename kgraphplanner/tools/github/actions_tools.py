"""
Agent-facing tools for github_actions_tool operations.

Phase 3: the CI-triage read set. The typical chain an agent follows is
list_workflow_runs -> list_run_jobs -> get_run_logs, narrowing from "what ran"
to "which job failed" to "what did it say", which is why the descriptions point
at each other.

Mutating operations (trigger, cancel, rerun) are gated server-side.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from kgraphplanner.tools.github.github_service_tool import GitHubServiceTool
from kgraphplanner.vital_agent_rest_resource_client.tools.github.actions_models import (
    GitHubActionsListWorkflowsInput, GitHubActionsListRunsInput,
    GitHubActionsGetRunInput, GitHubActionsListJobsInput, GitHubActionsRunLogsInput,
    GitHubActionsTriggerInput, GitHubActionsCancelRunInput, GitHubActionsRerunInput,
)

SERVICE_TOOL = "github_actions_tool"


def _run_summary(run) -> Dict[str, Any]:
    if run is None:
        return {}
    return {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "conclusion": run.conclusion,
        "branch": run.branch,
        "event": run.event,
        "actor": run.actor,
        "run_number": run.run_number,
        "run_attempt": run.run_attempt,
        "head_sha": run.head_sha,
        "created_at": run.created_at,
        # Exposed because the eval caught an agent presenting created_at as the
        # completion time: asked "did the last run pass?", it had only a start
        # timestamp and relabelled it. For a completed run updated_at is when it
        # finished, so the field it needed now exists rather than being inferred.
        "updated_at": run.updated_at,
        "url": run.html_url,
    }


class GitHubListWorkflowsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_workflows"
    TOOL_NAME = "github_list_workflows"
    DESCRIPTION = (
        "List the GitHub Actions workflows defined in a repository. "
        "Use this to find a workflow's id or filename before listing its runs."
    )
    INPUT_MODEL = GitHubActionsListWorkflowsInput
    AGENT_FIELDS = {
        "max_results": (Optional[int], Field(20, description="Maximum workflows to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "workflows": [{
                "id": w.id, "name": w.name, "path": w.path, "state": w.state, "url": w.html_url,
            } for w in output.workflows],
            **self.paging(output),
        }


class GitHubListWorkflowRunsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_workflow_runs"
    TOOL_NAME = "github_list_workflow_runs"
    DESCRIPTION = (
        "List GitHub Actions workflow runs, most recent first. Filter by branch, or by "
        "status='failure' to find failing builds. Each run has an id -- pass it to "
        "github_list_run_jobs to see which job failed."
    )
    INPUT_MODEL = GitHubActionsListRunsInput
    AGENT_FIELDS = {
        "workflow_id": (Optional[str], Field(None, description="Workflow id or filename to filter by")),
        "branch": (Optional[str], Field(None, description="Only runs on this branch")),
        "status": (Optional[str], Field(None, description="Filter by status or conclusion, e.g. failure, success")),
        "max_results": (Optional[int], Field(10, description="Maximum runs to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "runs": [_run_summary(r) for r in output.runs],
            "total_count": output.total_count,
            **self.paging(output),
        }


class GitHubGetWorkflowRunTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "get_workflow_run"
    TOOL_NAME = "github_get_workflow_run"
    DESCRIPTION = (
        "Get one GitHub Actions workflow run by id, with its status and conclusion. "
        "created_at is when the run started; updated_at is when it last changed, which for a completed run is when it finished."
    )
    INPUT_MODEL = GitHubActionsGetRunInput
    AGENT_FIELDS = {
        "run_id": (int, Field(..., description="Workflow run id", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        if output.run is None:
            return {"found": False}
        return {"found": True, **_run_summary(output.run)}


class GitHubListRunJobsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "list_run_jobs"
    TOOL_NAME = "github_list_run_jobs"
    DESCRIPTION = (
        "List the jobs in a workflow run, with each job's conclusion and per-step results. "
        "This is how to find which step of a failing build broke. For the actual error text, "
        "follow with github_get_run_logs."
    )
    INPUT_MODEL = GitHubActionsListJobsInput
    AGENT_FIELDS = {
        "run_id": (int, Field(..., description="Workflow run id", ge=1)),
        "max_results": (Optional[int], Field(20, description="Maximum jobs to return", ge=1, le=100)),
        "page": (Optional[int], Field(None, description="Pass the next_page value from a previous call", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        jobs = []
        for j in output.jobs:
            jobs.append({
                "id": j.id,
                "name": j.name,
                "status": j.status,
                "conclusion": j.conclusion,
                "started_at": j.started_at,
                "completed_at": j.completed_at,
                "url": j.html_url,
                # Only failing steps: a green job's step list is noise, and a
                # matrix build's full step list is a lot of it.
                "failed_steps": [
                    {"number": s.number, "name": s.name, "conclusion": s.conclusion}
                    for s in j.steps
                    if s.conclusion and s.conclusion not in ("success", "skipped")
                ],
                "step_count": len(j.steps),
            })
        return {"jobs": jobs, **self.paging(output)}


class GitHubGetRunLogsTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "get_run_logs"
    TOOL_NAME = "github_get_run_logs"
    DESCRIPTION = (
        "Get the tail of the log files for a workflow run. Raw CI logs run to megabytes, so "
        "only the last lines of each file are returned. Narrow with github_list_run_jobs "
        "first where possible, and raise max_lines_per_file only if the cause is not visible."
    )
    INPUT_MODEL = GitHubActionsRunLogsInput
    AGENT_FIELDS = {
        "run_id": (int, Field(..., description="Workflow run id", ge=1)),
        # Defaults deliberately below the service's: log text is the largest thing
        # any of these tools can return, and the agent pays for it in context.
        "max_lines_per_file": (Optional[int], Field(
            40, description="Tail lines kept per log file", ge=1, le=500)),
        "max_files": (Optional[int], Field(5, description="Maximum log files to include", ge=1, le=50)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "logs": [{
                "filename": log.filename,
                "lines": log.lines,
                "truncated": log.truncated,
            } for log in output.logs],
            "returned_count": output.returned_count,
            "truncated": output.truncated,
        }


ACTIONS_TOOLS = {
    cls.TOOL_NAME: cls for cls in (
        GitHubListWorkflowsTool,
        GitHubListWorkflowRunsTool,
        GitHubGetWorkflowRunTool,
        GitHubListRunJobsTool,
        GitHubGetRunLogsTool,
    )
}


# ---------------------------------------------------------------------------
# Writes
#
# trigger and rerun sit behind ALLOW_WORKFLOW_DISPATCH server-side, which is off
# by default. A denial arrives as an actionable api_error.
# ---------------------------------------------------------------------------

class GitHubTriggerWorkflowTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "trigger_workflow"
    TOOL_NAME = "github_trigger_workflow"
    MUTATING = True
    DESCRIPTION = (
        "Start a workflow_dispatch run on a branch or tag. Only works for workflows that "
        "declare a workflow_dispatch trigger, and is disabled by policy in most deployments. "
        "GitHub returns no run id for a dispatch, so the response explains how the run was "
        "located if it was."
    )
    INPUT_MODEL = GitHubActionsTriggerInput
    AGENT_FIELDS = {
        "workflow_id": (str, Field(..., description="Workflow id or filename", min_length=1)),
        "ref": (str, Field(..., description="Branch or tag to run against", min_length=1)),
        "inputs": (Optional[dict], Field(None, description="Inputs declared by the workflow")),
    }

    def project(self, output) -> Dict[str, Any]:
        return {
            "triggered": bool(output.triggered),
            "note": output.dispatch_note,
            "run": _run_summary(output.run) if output.run else None,
        }


class GitHubCancelWorkflowRunTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "cancel_workflow_run"
    TOOL_NAME = "github_cancel_workflow_run"
    MUTATING = True
    DESCRIPTION = (
        "Cancel a workflow run that is queued or in progress. Cancelling a completed run "
        "does nothing."
    )
    INPUT_MODEL = GitHubActionsCancelRunInput
    AGENT_FIELDS = {
        "run_id": (int, Field(..., description="Workflow run id", ge=1)),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"cancelled": True, "run_id": getattr(output.run, "id", None),
                "run": _run_summary(output.run) if output.run else None}


class GitHubRerunWorkflowTool(GitHubServiceTool):
    SERVICE_TOOL = SERVICE_TOOL
    OPERATION = "rerun_workflow"
    TOOL_NAME = "github_rerun_workflow"
    MUTATING = True
    DESCRIPTION = (
        "Re-run a completed workflow run, optionally only its failed jobs. Useful for "
        "confirming a flaky failure, but note it creates permanent run history and is "
        "disabled by policy in most deployments."
    )
    INPUT_MODEL = GitHubActionsRerunInput
    AGENT_FIELDS = {
        "run_id": (int, Field(..., description="Workflow run id", ge=1)),
        "failed_jobs_only": (Optional[bool], Field(
            False, description="Re-run only the jobs that failed")),
    }

    def project(self, output) -> Dict[str, Any]:
        return {"rerun_started": bool(output.triggered), "note": output.dispatch_note,
                "run": _run_summary(output.run) if output.run else None}


ACTIONS_TOOLS.update({
    cls.TOOL_NAME: cls for cls in (
        GitHubTriggerWorkflowTool,
        GitHubCancelWorkflowRunTool,
        GitHubRerunWorkflowTool,
    )
})
