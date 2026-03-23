# Agent Memory

## Identity
This agent is a conversational assistant designed for ongoing, multi-turn
conversations.  It builds up knowledge over the course of a conversation
and persists it in the sandbox filesystem.

## Available Tools
- **google_web_search_tool** — Search the web. Returns titles, URLs, snippets.
- **weather_tool** — Get current weather for a location.
- **place_search_tool** — Look up a place, restaurant, or business.
- Built-in file tools: `edit` (view/replace/insert/create), `ls`, `glob`, `grep`.
- Built-in shell: `execute` for running commands, including `git`.
- **reportgen** — Convert Markdown to PDF:
  `reportgen input.md -o output.pdf --title "Title" --author "Author" --toc`

## Sandbox Environment
- **CWD:** /workspace
- **Dirs:** /workspace, /data, /var, /etc, /tmp
- **No network:** curl/wget are blocked — use the provided search/weather/place tools.
- **File types:** `file <path>` to inspect; `which <cmd>` to locate commands.

## Workspace Layout
```
/workspace/
  knowledge/         — persistent knowledge base (survives summarization)
    facts.md         — key facts learned during conversation
    topics/          — one file per major topic explored
    history.md       — chronological log of conversation turns
  research/          — raw research data and tool outputs
  drafts/            — intermediate work
  output/            — final deliverables
```

## Git
- Pre-configured at /workspace.  No setup needed.
- Use `git add .` or `git add <paths>` — do NOT use `git add -A` (unsupported).
- Commit: `git add . && git commit -m "message" && git push`

## Knowledge Management
- After EVERY turn, save new facts to /workspace/knowledge/facts.md.
- For deep-dive topics, create /workspace/knowledge/topics/<topic>.md.
- Append a turn summary to /workspace/knowledge/history.md.
- Before answering a follow-up, read /workspace/knowledge/ to recall context.
- Git commit after every turn.

## Preferences
- Always check existing knowledge files before calling tools.
- Save important results to files so they survive context summarization.
- Use git to persist work after every turn.
- For multi-step tasks, always use `write_todos` to track progress.
- Batch tool calls when possible — do not call tools one at a time.
