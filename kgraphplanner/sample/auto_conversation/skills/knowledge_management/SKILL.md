# Knowledge Management Skill

When managing knowledge across conversation turns, follow this procedure:

## After Every Turn
1. Identify new facts, decisions, or research results from this turn.
2. Append them to `/workspace/knowledge/facts.md` (create if missing).
3. If a new topic was explored in depth, create or update
   `/workspace/knowledge/topics/<topic>.md`.
4. Append a one-line turn summary to `/workspace/knowledge/history.md`:
   ```
   - Turn N: <brief summary of what was discussed/learned>
   ```
5. Git commit: `git add . && git commit -m "Turn N: <summary>" && git push`
   NOTE: Do NOT use `git add -A` — the sandbox does not support the -A flag.

## Before Answering Follow-ups
1. Read `/workspace/knowledge/facts.md` to recall previously learned facts.
2. Check `/workspace/knowledge/topics/` for relevant topic files.
3. Read `/workspace/knowledge/history.md` to understand conversation flow.
4. Only call tools if the needed information is NOT already in knowledge files.

## File Formats
- `facts.md` — bullet list of key facts, grouped by topic.
- `topics/<name>.md` — deeper notes on a specific topic with sources.
- `history.md` — chronological list of turn summaries.

## Rules
- Never duplicate information already in the knowledge base.
- Update existing entries rather than appending duplicates.
- Always cite sources (URLs) when saving web search results.
- Keep files concise — summarize, don't copy raw tool output verbatim.
