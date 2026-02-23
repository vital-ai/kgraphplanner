"""Deep Agents Example: Editorial Pipeline with edit, outline, and reportgen.

Two-agent editorial workflow using AgentBox Tier 1/3 commands:

  1. Upload a draft Markdown file into the sandbox via the API.
  2. Agent 1 ("Junior Editor") revises the content using ``edit`` (str_replace)
     to fix spelling, grammar, and style issues.
  3. Agent 2 ("Senior Editor") further polishes the content, adds structure,
     and runs ``reportgen`` to generate a PDF report.
  4. Download the finished files from the sandbox via the API.

Demonstrates:
    - upload_files / download_files: host ↔ MemFS file transfer
    - edit (Tier 1): str_replace for targeted text edits
    - outline (Tier 3): structural overview of document
    - reportgen (Tier 3): pandoc-based PDF generation

Prerequisites:
    docker compose up --build -d
    pip install deepagents langchain-openai
    OPENAI_API_KEY in .env
    pandoc + LaTeX installed on worker (for reportgen)

Usage:
    python examples/deep_agents_editorial.py
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*NotRequired.*", category=UserWarning)

from dotenv import load_dotenv

load_dotenv()

from deepagents import create_deep_agent  # noqa: E402
from agentbox.deepagents import AgentBoxSandbox  # noqa: E402


ORCHESTRATOR = "http://localhost:8090"

# Paths on the host filesystem (must be within AGENTBOX_BOXCP_LOCAL_ALLOW)
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_MD = os.path.join(EXAMPLES_DIR, "data", "draft_report.md")
OUTPUT_DIR = os.path.join(EXAMPLES_DIR, "data", "output")


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_agent(agent, task, label):
    """Invoke a Deep Agent with a task and print the result."""
    print_header(label)
    print(f"  Task: {task[:120]}{'...' if len(task) > 120 else ''}\n")

    result = agent.invoke({
        "messages": [{"role": "user", "content": task}],
    })

    # Print the final assistant message
    messages = result.get("messages", [])
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if role == "ai" and content:
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


def main():
    # Verify input file exists
    if not os.path.exists(INPUT_MD):
        print(f"Error: Input file not found: {INPUT_MD}")
        print("  Create examples/data/draft_report.md first.")
        return False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set up log file
    log_dir = os.path.join(EXAMPLES_DIR, "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"editorial_{ts}.log")
    print(f"  Logging to {os.path.abspath(log_file)}")

    print_header("EDITORIAL PIPELINE: Draft → Junior Edit → Senior Edit → PDF")

    # ---- Create sandbox (MemBox — no git needed) ----
    backend = AgentBoxSandbox.create(
        ORCHESTRATOR, box_type="mem", log_file=log_file,
    )
    time.sleep(2)

    # ---- Step 1: Upload draft into sandbox ----
    print_header("STEP 1: Upload draft into sandbox")
    with open(INPUT_MD, "rb") as f:
        draft_content = f.read()
    responses = backend.upload_files([("/draft.md", draft_content)])
    for r in responses:
        status = "✓" if r.error is None else f"✗ {r.error}"
        print(f"  {r.path}: {status}")

    # Show the draft
    result = backend.execute("edit /draft.md --info")
    print(f"  Info: {result.output}")

    result = backend.execute("outline /draft.md")
    print(f"  Outline:\n{result.output}")

    # ---- Step 2: Junior Editor ----
    agent1 = create_deep_agent(
        backend=backend,
        system_prompt=(
            "You are a junior editor. You fix spelling, grammar, and "
            "punctuation errors in documents. You work in a sandbox with "
            "these shell commands:\n"
            "  edit <file> --view [--range N:M]  — view file with line numbers\n"
            "  edit <file> --old '...' --new '...'  — find and replace text\n"
            "  edit <file> --info  — file metadata and structure\n"
            "  outline <file>  — structural outline of definitions/headings\n\n"
            "Use 'edit --old/--new' for each correction. Make targeted fixes, "
            "not bulk rewrites. Show your work by fixing errors one at a time."
        ),
    )

    run_agent(
        agent1,
        task=(
            "Review the file /draft.md. It is a quarterly financial report "
            "with many spelling and grammar errors. Your job:\n\n"
            "1. First, view the file to understand its contents.\n"
            "2. Find and fix ALL spelling errors (there are many — "
            "   'revenu' → 'revenue', 'summerizes' → 'summarizes', "
            "   'expences' → 'expenses', 'proffit' → 'profit', etc.)\n"
            "3. Fix grammar issues ('companys' → 'company's', etc.)\n"
            "4. Fix the title: 'Reveneu' → 'Revenue'\n"
            "5. Fix 'Exective' → 'Executive'\n\n"
            "Use edit --old/--new for each fix. After all fixes, view the "
            "final file to confirm it reads cleanly."
        ),
        label="AGENT 1: JUNIOR EDITOR — Fix Spelling & Grammar",
    )

    # ---- Step 3: Senior Editor ----
    agent2 = create_deep_agent(
        backend=backend,
        system_prompt=(
            "You are a senior editor and report designer. You polish "
            "documents for executive audiences and generate professional "
            "PDF reports.\n\n"
            "IMPORTANT: You work ONLY via shell commands. Do NOT write Python code. "
            "Do NOT try to install packages. Use these commands:\n\n"
            "  edit <file> --view [--range N:M]  — view file with line numbers\n"
            "  edit <file> --old '...' --new '...'  — find and replace text\n"
            "  edit <file> --info  — file metadata and structure\n"
            "  outline <file>  — structural outline of the document\n"
            "  reportgen <file> -o <output.pdf> [options]  — generate PDF\n\n"
            "reportgen is a SHELL COMMAND (not a Python library). Example:\n"
            "  reportgen /draft.md -o /report.pdf --title 'My Report' --toc\n\n"
            "reportgen options: --title, --author, --date, --toc, --toc-depth, "
            "--highlight-style, --margin\n\n"
            "Use 'edit --old/--new' for content changes, then 'reportgen' to "
            "produce the final PDF."
        ),
    )

    run_agent(
        agent2,
        task=(
            "You are working with /draft.md, which has already been "
            "spell-checked by a junior editor. Your job:\n\n"
            "1. View the file and outline to assess the current state.\n"
            "2. Improve the Executive Summary — make it more concise and "
            "   impactful for a C-suite audience.\n"
            "3. Add a brief 'Conclusion' section at the end summarizing "
            "   the key takeaway and recommended next steps.\n"
            "4. Ensure the tone is professional and consistent throughout.\n"
            "5. Generate a PDF report:\n"
            "   reportgen /draft.md -o /report.pdf "
            "--title 'Q4 2025 Quarterly Report' "
            "--author 'Finance Team' "
            "--date '2026-01-15' --toc\n"
            "6. Confirm the PDF was generated successfully."
        ),
        label="AGENT 2: SENIOR EDITOR — Polish & Generate PDF",
    )

    # ---- Step 4: Download results from sandbox ----
    print_header("STEP 4: Download results from sandbox")

    downloads = backend.download_files(["/draft.md", "/report.pdf"])
    for dl in downloads:
        if dl.error:
            print(f"  ✗ {dl.path}: {dl.error}")
            if dl.path == "/report.pdf":
                print("    (pandoc may not be installed on the worker)")
            continue

        # Write to output directory
        filename = os.path.basename(dl.path)
        if filename == "draft.md":
            filename = "final_report.md"
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "wb") as f:
            f.write(dl.content)
        print(f"  ✓ {dl.path} → {out_path} ({len(dl.content):,} bytes)")

    # ---- Show final state ----
    print_header("FINAL DOCUMENT")
    result = backend.execute("edit /draft.md --view")
    lines = result.output.strip().split("\n")
    for line in lines[:40]:
        print(f"    {line}")
    if len(lines) > 40:
        print(f"    ... ({len(lines) - 40} more lines)")

    backend.destroy()
    print_header("EDITORIAL PIPELINE: Complete ✓")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
