# Research Skill

When performing research tasks, follow this procedure:

## Planning
1. Break the user's request into discrete research questions.
2. Write a todo list (`write_todos`) with one item per question.
3. Identify which tool is best for each question:
   - `google_web_search_tool` — general web search, news, facts, comparisons
   - `weather_tool` — current weather for a specific location
   - `place_search_tool` — find a place, restaurant, or business by name/type

## Execution
4. Call tools for as many questions as possible in a single turn.
   - If multiple searches are needed, call them all at once — do NOT
     serialize them one by one.
5. Save large or important results to files in `/workspace/research/`
   so they remain available if the conversation is summarized.
6. After each batch of tool calls, update the todo list to track progress.

## Synthesis
7. Once all data is gathered, compile a clear, well-structured answer.
8. Cite sources (URLs) when available from web search results.
9. If the user asked for a written report, save it to `/workspace/output/`
   and mention the file path.

## Rules
- Never fabricate data — if a tool returns no results, say so.
- If a query is ambiguous, prefer a broader search first, then refine.
- For multi-step tasks (e.g. "find restaurants then check weather"),
  complete one phase before starting the next.
