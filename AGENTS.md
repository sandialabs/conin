# AGENTS.md

## Development Guidelines

- Keep changes minimal and focused.
- Prefer stable public APIs when practical. If APIs change, update all affected
  code, tests, examples, and documentation together.
- Do not revert unrelated user or collaborator changes.
- Public examples should live in each model package's `examples.py` module, not under test modules.

## Documentation

- Update docs when code changes affect public behavior, public APIs, examples, optional dependencies, installation, or backend requirements.
- User-facing docs live in `doc/`; Sphinx config is `doc/conf.py`; Read the Docs config is `.readthedocs.yaml`.
- Keep short backend-free examples runnable when practical, preferably with Sphinx doctest blocks.

## Testing

- New behavior should include tests or updates to existing tests.
- Run targeted tests for the code you change. Use the full suite when feasible with `pytest`.
- For docs changes, run `python -m sphinx -b doctest doc doc/_build/doctest` and `python -m sphinx -b html doc doc/_build/html`.
- Remove generated `doc/_build` output before committing.
