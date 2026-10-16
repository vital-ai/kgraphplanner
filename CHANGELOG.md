# Changelog

The exec graph's release notes for 0.2.0 and 0.2.1, including the migration
notes, are in [docs/exec_graph.md](docs/exec_graph.md).

## 0.2.2 (2026-10-16)

### Tool client keeps every field the tool service returns

The tool client's models and handlers had fallen behind the tool service
(`vital-agent-resource-rest`). Pydantic ignores undeclared fields, so these
values were dropped silently before they reached the agent.

- **Serper results** now carry `category`, `phone_number`, `website`,
  `latitude`, `longitude`, `price_level` and `sitelinks`.
  - **`phone`:** it was always `None`, because the service sends
    `phone_number`. It's now a deprecated alias with the same value. It's
    left out of `model_dump()`, so the phone appears once in what the model
    sees.
- **The Serper Knowledge Graph** now carries `website` and `image_url`.
- **Serper search input** accepts `language`, `country` and `page`, and
  `serper_web_search_tool` passes them through.
- **Place search results** now carry `rating` and `user_ratings_total`. This
  needs tool service `61f1900` or later; older services leave them `None`.
  Results past the service's Place Details limit (default 3) have no phone,
  website, Maps URL or vicinity.
- **Handlers build results from the whole wire dict** (the new
  `build_model`), so a field added to a model reaches the agent without a
  handler change. An optional field whose value doesn't validate is dropped
  and logged by name, instead of failing the result.
- **A parity test** (`tests/test_tool_client_parity.py`) checks each client
  model against a recorded snapshot of the service's fields,
  `tests/data/tool_service_fields.json`. Regenerate the snapshot after a
  service change:

  ```
  python scripts/snapshot_tool_service_fields.py ../vital-agent-resource-rest
  ```

- **Removed:** the unused place search and weather handler modules
  (`place_search_tool_handler.py`, `place_search_response.py`,
  `weather_tool_handler.py`). Nothing imported them, and the registered
  handlers are each package's `tool_handler.py`.
