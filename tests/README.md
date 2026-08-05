# tests/

Pytest suite. **Offline and deterministic** — no tool service, no Weaviate, no
model calls, no network. `pytest` from the repo root should pass on a laptop with
nothing else running.

Anything needing a live dependency is either:

- a case script under `test_scripts*/`, run by hand against a running stack, or
- marked `@pytest.mark.integration` here and deselected with `-m 'not integration'`.

The ad-hoc agent scripts that used to live in `test/` are now in
`test_scripts_agents/` — they drive real agents against real models and are not
pytest tests.

Run:

    pytest                      # everything offline
    pytest -m 'not integration' # explicit, same thing today
    pytest tests/test_agent_config.py -v
