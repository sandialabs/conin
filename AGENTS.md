# AGENTS.md

## Development Guidelines

- Keep changes minimal and focused.
- Prefer stable public APIs when practical. If APIs change, update all affected
  code, tests, examples, and documentation together.
- Do not revert unrelated user or collaborator changes.
- Public examples should live in `conin/examples/` (`markov_network.py`, `bayesian_network.py`, `dynamic_bayesian_network.py`, `hidden_markov_model.py`), not under test modules.
- The package layout, the import layering rules and the design decisions (no representation cache, VE through pgmpy, the representation registry) are in `doc/sphinx/architecture.rst`. Follow its layering rules; new backends or constraint families are added by registering representation builders.
- There are no compatibility aliases for moved names. If a public name moves, update every importer (code, tests, docs, notebooks, `doc/notebooks/MVR/*.md`) together.
- `import conin` must not import torch; the MVR packages load their algorithms lazily.

## Documentation

- Update docs when code changes affect public behavior, public APIs, examples, optional dependencies, installation, or backend requirements.
- User-facing docs live in `doc/sphinx/`; Sphinx config is `doc/sphinx/conf.py`; Read the Docs config is `.readthedocs.yaml`.
- Every Python example in `doc/sphinx/` must be a `testcode` or `doctest` block (not `code-block:: python`), so the CI `docs` job runs it. Examples that need pgmpy, pytoulbar2 or a MIP solver add `:skipif: not pgmpy_available` (or `not toulbar2_available`, `not mip_solver_available`; defined in `doctest_global_setup` in `doc/sphinx/conf.py`). No example may need torch.

## Testing

- New behavior should include tests or updates to existing tests.
- Run targeted tests for the code you change. Use the full suite when feasible with `pytest`.
- For docs changes, run `python -m sphinx -b doctest doc/sphinx doc/sphinx/_build/doctest` and `python -m sphinx -W --keep-going -b html doc/sphinx doc/sphinx/_build/html` (CI runs both, with warnings as errors, and with `CONIN_DOCS_REQUIRE_BACKENDS=1`, so a missing backend fails instead of skipping examples).
- Remove generated `doc/sphinx/_build` output before committing.
