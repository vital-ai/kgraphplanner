from typing import Optional, Union, Any
from pydantic import BaseModel, Field
from .tool_name_enum import ToolName
from .google_address_validation.models import AddressValidationInput
from .place_search.models import PlaceSearchInput
from .weather.models import WeatherInput
from .serper_web_search.models import SerperWebSearchInput
from .web_search.models import WebSearchInput
from .github.issue_models import (
    GitHubIssueListLabelsInput, GitHubIssueListMilestonesInput,
    GitHubIssueListAssignableUsersInput, GitHubIssueFindByBodyInput,
    GitHubIssueGetInput,
    GitHubIssueListInput,
    GitHubIssueCommentListInput,
    GitHubIssueSearchInput,
    GitHubIssueCreateInput, GitHubIssueUpdateInput, GitHubIssueCloseInput,
    GitHubIssueReopenInput, GitHubIssueCommentCreateInput,
    GitHubIssueCommentUpdateInput, GitHubIssueCommentDeleteInput,
    GitHubIssueAddLabelsInput, GitHubIssueRemoveLabelsInput,
    GitHubIssueAddAssigneesInput, GitHubIssueRemoveAssigneesInput,
)
from .github.pr_models import (
    GitHubPRListInput,
    GitHubPRGetInput,
    GitHubPRFilesInput,
    GitHubPRCommentListInput,
    GitHubPRReviewListInput,
    GitHubPRCreateInput, GitHubPRUpdateInput, GitHubPRCommentCreateInput,
    GitHubPRReviewCreateInput, GitHubPRRequestReviewersInput,
)
from .github.code_models import (
    GitHubCreateBranchInput, GitHubDeleteBranchInput, GitHubWriteFileInput,
    GitHubDeleteFileInput, GitHubMergeInput, GitHubWriteFilesInput,
)
from .github.repo_models import (
    GitHubRepoGetInput, GitHubGetFileInput, GitHubListBranchesInput,
    GitHubListCommitsInput, GitHubGetCommitInput, GitHubCompareRefsInput,
    GitHubGetAuthenticatedUserInput,
)
from .github.actions_models import (
    GitHubActionsListWorkflowsInput,
    GitHubActionsListRunsInput,
    GitHubActionsGetRunInput,
    GitHubActionsListJobsInput,
    GitHubActionsRunLogsInput,
    GitHubActionsTriggerInput, GitHubActionsCancelRunInput, GitHubActionsRerunInput,
)


class ToolRequest(BaseModel):
    """Base tool request model with non-tool-specific parameters"""
    tool: ToolName = Field(..., description="Tool name to execute")
    request_id: Optional[str] = Field(None, description="Optional request identifier")
    timeout: Optional[int] = Field(None, description="Request timeout in seconds")
    # The GitHub inputs carry a Literal `operation` discriminator, so pydantic's
    # smart union matches them exactly rather than by field overlap. Verified in
    # test_scripts/cases/case_github_phase0.py -- see the plan's section 9.1.
    tool_input: Union[
        AddressValidationInput,
        PlaceSearchInput,
        SerperWebSearchInput,
        WeatherInput,
        WebSearchInput,
        GitHubIssueGetInput,
        GitHubIssueListInput,
        GitHubIssueCommentListInput,
        GitHubIssueSearchInput,
        GitHubPRListInput,
        GitHubPRGetInput,
        GitHubPRFilesInput,
        GitHubPRCommentListInput,
        GitHubPRReviewListInput,
        GitHubActionsListWorkflowsInput,
        GitHubActionsListRunsInput,
        GitHubActionsGetRunInput,
        GitHubActionsListJobsInput,
        GitHubActionsRunLogsInput,
        GitHubIssueCreateInput,
        GitHubIssueUpdateInput,
        GitHubIssueCloseInput,
        GitHubIssueReopenInput,
        GitHubIssueCommentCreateInput,
        GitHubIssueCommentUpdateInput,
        GitHubIssueCommentDeleteInput,
        GitHubIssueAddLabelsInput,
        GitHubIssueRemoveLabelsInput,
        GitHubIssueAddAssigneesInput,
        GitHubIssueRemoveAssigneesInput,
        GitHubIssueListLabelsInput,
        GitHubIssueListMilestonesInput,
        GitHubIssueListAssignableUsersInput, GitHubIssueFindByBodyInput,
        GitHubPRCreateInput,
        GitHubPRUpdateInput,
        GitHubPRCommentCreateInput,
        GitHubPRReviewCreateInput,
        GitHubPRRequestReviewersInput,
        GitHubActionsTriggerInput,
        GitHubActionsCancelRunInput,
        GitHubActionsRerunInput,
        GitHubCreateBranchInput,
        GitHubDeleteBranchInput,
        GitHubWriteFileInput,
        GitHubDeleteFileInput,
        GitHubMergeInput,
        GitHubWriteFilesInput,
        GitHubRepoGetInput,
        GitHubGetFileInput,
        GitHubListBranchesInput,
        GitHubListCommitsInput,
        GitHubGetCommitInput,
        GitHubCompareRefsInput,
        GitHubGetAuthenticatedUserInput
        ] = Field(..., description="Tool-specific input parameters")

    model_config = {"extra": "allow"}
    
    def to_dict(self) -> dict:
        """
        Converts ToolRequest instance to a dictionary.
        
        Returns:
            dict: The dictionary representation of the tool request.
        """
        return self.model_dump(exclude_none=True)
