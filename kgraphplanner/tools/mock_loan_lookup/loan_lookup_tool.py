"""
Mock Loan Lookup Tool — returns deterministic fake account data.

Used by the sales agent test to exercise KGraphToolWorker in a
chat↔tool loop without requiring an external loan-management API.

Mock data rules:
  - Emails containing "john" or "acme"  → active application (APP-20240115)
  - Emails containing "jane" or "widgets" → active loan (LN-98765)
  - Loan IDs starting with "LN-"         → active loan match
  - Loan IDs starting with "APP-"        → active application match
  - Phone numbers                        → active loan (generic)
  - Anything else                        → not found
"""

import json
import logging
from typing import Callable, Type, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from kgraphplanner.tool_manager.tool_inf import AbstractTool

logger = logging.getLogger(__name__)

TOOL_NAME = "loan_lookup_tool"


class LoanLookupInput(BaseModel):
    """Input schema for the loan lookup tool."""
    identifier_type: str = Field(
        description="Type of identifier: 'email', 'phone', 'loan_id', 'application_id', or 'name'"
    )
    identifier_value: str = Field(
        description="The identifier value to look up (e.g. 'john@acme.com', 'LN-98765')"
    )


# ── Mock database ──────────────────────────────────────────────

MOCK_ACCOUNTS: Dict[str, Dict[str, Any]] = {
    "john@acme.com": {
        "found": True,
        "account_type": "application",
        "application_id": "APP-20240115",
        "email": "john@acme.com",
        "name": "John Smith",
        "status": "under_review",
        "submitted_date": "2024-01-15",
        "loan_amount_requested": "$150,000",
        "business_name": "Acme Corp",
        "next_step": "Underwriting review — estimated 2-3 business days",
    },
    "jane@widgets.co": {
        "found": True,
        "account_type": "loan",
        "loan_id": "LN-98765",
        "email": "jane@widgets.co",
        "name": "Jane Doe",
        "status": "active",
        "balance": "$47,250.00",
        "next_payment_date": "2024-02-15",
        "next_payment_amount": "$1,875.00",
        "interest_rate": "7.5%",
        "business_name": "Widgets Co",
    },
    "LN-98765": {
        "found": True,
        "account_type": "loan",
        "loan_id": "LN-98765",
        "email": "jane@widgets.co",
        "name": "Jane Doe",
        "status": "active",
        "balance": "$47,250.00",
        "next_payment_date": "2024-02-15",
        "next_payment_amount": "$1,875.00",
        "interest_rate": "7.5%",
        "business_name": "Widgets Co",
    },
    "APP-20240115": {
        "found": True,
        "account_type": "application",
        "application_id": "APP-20240115",
        "email": "john@acme.com",
        "name": "John Smith",
        "status": "under_review",
        "submitted_date": "2024-01-15",
        "loan_amount_requested": "$150,000",
        "business_name": "Acme Corp",
        "next_step": "Underwriting review — estimated 2-3 business days",
    },
}


def _lookup(identifier_type: str, identifier_value: str) -> Dict[str, Any]:
    """Deterministic mock lookup logic."""
    value = identifier_value.strip()

    # Direct match
    if value in MOCK_ACCOUNTS:
        return MOCK_ACCOUNTS[value]

    # Fuzzy email match (john/acme → application, jane/widgets → loan)
    lower = value.lower()
    if identifier_type == "email":
        if "john" in lower or "acme" in lower:
            return MOCK_ACCOUNTS["john@acme.com"]
        if "jane" in lower or "widget" in lower:
            return MOCK_ACCOUNTS["jane@widgets.co"]

    # Loan/application ID prefix match
    if identifier_type in ("loan_id", "application_id") or value.startswith(("LN-", "APP-")):
        if value.startswith("LN-"):
            return {**MOCK_ACCOUNTS["LN-98765"], "loan_id": value}
        if value.startswith("APP-"):
            return {**MOCK_ACCOUNTS["APP-20240115"], "application_id": value}

    # Phone → generic active loan
    if identifier_type == "phone":
        return {
            "found": True,
            "account_type": "loan",
            "loan_id": "LN-55555",
            "phone": value,
            "name": "Phone Customer",
            "status": "active",
            "balance": "$32,100.00",
            "next_payment_date": "2024-03-01",
            "next_payment_amount": "$1,200.00",
        }

    # Not found
    return {
        "found": False,
        "identifier_type": identifier_type,
        "identifier_value": value,
        "reason": "No matching account found for the provided identifier.",
    }


class LoanLookupTool(AbstractTool):
    """Mock loan-account lookup tool for testing."""

    def __init__(self, config=None, tool_manager=None):
        super().__init__(
            config=config or {},
            tool_manager=tool_manager,
            name=TOOL_NAME,
            description="Look up a business-loan account or application by email, phone, loan ID, or application ID. Returns account details if found.",
        )

    def get_tool_schema(self) -> Type[BaseModel]:
        return LoanLookupInput

    def get_tool_function(self) -> Callable:

        @tool(args_schema=LoanLookupInput)
        def loan_lookup_tool(identifier_type: str, identifier_value: str) -> str:
            """Look up a business-loan account or application by identifier.

            Args:
                identifier_type: Type of identifier — 'email', 'phone', 'loan_id', 'application_id', or 'name'.
                identifier_value: The value to search for (e.g. 'john@acme.com', 'LN-98765').

            Returns:
                JSON string with account details or a not-found message.
            """
            logger.info(f"🔍 loan_lookup_tool: type={identifier_type} value={identifier_value}")
            result = _lookup(identifier_type, identifier_value)
            logger.info(f"📋 loan_lookup_tool: found={result.get('found')}")
            return json.dumps(result, indent=2)

        return loan_lookup_tool
