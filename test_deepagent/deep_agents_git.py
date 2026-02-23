"""Deep Agents Example: LLM-driven coding agent with Git persistence.

Uses LangChain Deep Agents (``create_deep_agent``) with an AgentBox
sandbox backend. The LLM decides what code to write, what commands to
run, and when to commit. Two agents collaborate through a shared Git repo:

  Agent 1 ("Architect") — receives a task, writes code, runs tests,
  commits and pushes. All decisions made by the LLM.

  Agent 2 ("Reviewer") — picks up the repo in a new sandbox (auto-restored
  from S3), reviews code, adds improvements, tests, commits and pushes.

Prerequisites:
    docker compose up --build -d
    pip install deepagents langchain-openai
    OPENAI_API_KEY in .env

Usage:
    python examples/deep_agents_git.py
"""

import os
import sys
import time
import uuid
import warnings

warnings.filterwarnings("ignore", message=".*NotRequired.*", category=UserWarning)

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv

load_dotenv()

from deepagents import create_deep_agent  # noqa: E402
from agentbox.deepagents import AgentBoxSandbox  # noqa: E402


ORCHESTRATOR = "http://localhost:8090"
MINIO_ENDPOINT = "http://localhost:9100"
MINIO_BUCKET = "agentbox-repos"
S3_PREFIX = "repos/"


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_agent(agent, task, label):
    """Invoke a Deep Agent with a task and print the result."""
    print_header(label)
    print(f"  Task: {task}\n")

    result = agent.invoke({
        "messages": [{"role": "user", "content": task}],
    })

    # Print the final assistant message
    messages = result.get("messages", [])
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if role == "ai" and content:
            # content may be a string or a list of content blocks
            if isinstance(content, list):
                text = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
            else:
                text = str(content)
            if text.strip():
                print(f"\n  [Agent Response]\n")
                for line in text.strip().split("\n"):
                    print(f"    {line}")
    print()
    return result


def cleanup_s3(repo_id):
    """Remove all S3 objects for a repo."""
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=MINIO_BUCKET, Prefix=f"{S3_PREFIX}{repo_id}/"):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=MINIO_BUCKET, Key=obj["Key"])


def main():
    repo_id = f"deep-agents-{uuid.uuid4().hex[:8]}"

    # Set up log file
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"deep_agents_{ts}_{repo_id}.log")
    print(f"  Logging to {os.path.abspath(log_file)}")

    print_header(f"DEEP AGENTS + GIT SANDBOX (repo_id={repo_id})")

    # ---- Agent 1: Architect ----
    backend1 = AgentBoxSandbox.create(
        ORCHESTRATOR, box_type="git", repo_id=repo_id, log_file=log_file,
    )
    time.sleep(2)  # wait for sandbox init

    agent1 = create_deep_agent(
        backend=backend1,
        system_prompt=(
            "You are a Python developer working in /workspace. "
            "You have access to a git repo backed by persistent storage. "
            "Use shell commands to write files, run code, and use git. "
            "When done, git add, git commit, and git push your work."
        ),
    )

    run_agent(
        agent1,
        task=(
            "Create a Python calculator module at /workspace/calculator.py with "
            "functions: add, subtract, multiply, divide (handle division by zero). "
            "Then create /workspace/test_calculator.py with unit tests for each function. "
            "Run the tests to verify they pass. "
            "Then git add both files, commit with a descriptive message, and git push."
        ),
        label="AGENT 1: ARCHITECT — Build & Test",
    )

    backend1.destroy()
    print("  Work persisted in S3.\n")

    # ---- Agent 2: Reviewer ----
    backend2 = AgentBoxSandbox.create(
        ORCHESTRATOR, box_type="git", repo_id=repo_id, log_file=log_file,
    )
    time.sleep(2)

    agent2 = create_deep_agent(
        backend=backend2,
        system_prompt=(
            "You are a senior Python developer reviewing code in /workspace. "
            "The repo was auto-restored from storage — check git log to see history. "
            "You have access to a git repo backed by persistent storage. "
            "Use shell commands to read files, write files, run code, and use git. "
            "When done, git add, git commit, and git push your improvements."
        ),
    )

    run_agent(
        agent2,
        task=(
            "Review the existing calculator module and tests in /workspace. "
            "Add two new functions to calculator.py: power(base, exp) and modulo(a, b) "
            "(handle modulo by zero). Add tests for both new functions to test_calculator.py. "
            "Run all the tests to make sure everything passes. "
            "Then git add, commit with a descriptive message, and git push."
        ),
        label="AGENT 2: REVIEWER — Extend & Improve",
    )

    # Show final state
    print_header("FINAL STATE")
    result = backend2.execute("cd /workspace && git log --oneline")
    print(f"  Git log:\n{result.output}\n")
    result = backend2.execute("cd /workspace && cat calculator.py")
    print(f"  calculator.py:\n{result.output}\n")

    backend2.destroy()

    # Cleanup
    cleanup_s3(repo_id)
    print_header("DEEP AGENTS + GIT SANDBOX: Complete ✓")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
