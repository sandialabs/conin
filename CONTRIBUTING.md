Contributing to Conin
=====================

Contributions are welcome, from inside and outside Sandia: bug reports,
questions, documentation fixes, examples, tests and new features. This file
explains how to get started and what reviewers look for. If anything here is
unclear, open an issue and ask.

Ways to Contribute
------------------

- **Report a bug or ask a question** in the
  [issue tracker](https://github.com/sandialabs/conin/issues). For a bug,
  include a small example that reproduces it, the conin version, and the
  backends involved (MIP solver, toulbar2, torch).
- **Fix something small** (a typo, a docstring, a clear bug) by opening a
  pull request directly.
- **Propose a larger change** (a new feature, backend, constraint family or
  model type, or a change to the public API) in an issue first, so we can
  agree on the approach before you put time into it.

You don't need to get everything right the first time. Open a draft pull
request early if you want feedback, and maintainers will help with failing
CI checks and the conventions below.

Development Setup
-----------------

```
git clone https://github.com/sandialabs/conin.git
cd conin
conda env create -f dev_environment.yml
conda activate conin_dev
python -m pip install -e '.[docs,test]'
nbstripout --install --attributes .gitattributes
python -m pytest conin
```

The optional backends are extras (``.[pgmpy]``, ``.[toulbar2]``,
``.[torch]``, ``.[regex]``; see ``README.md``). Tests that need a backend
that isn't installed are skipped, so you only need the ones your change
touches. CI checks formatting with ``black`` and lint with ``flake8``.
``flake8`` is in the dev environment; install ``black`` with
``python -m pip install black``, and run both before you push.

Using AI Tools
--------------

You may use AI coding assistants to contribute. The repository supports
them: ``AGENTS.md`` gives agents the project's context, and the ``/pr-check``
Claude Code skill checks a branch against this file. Whatever tools you use,
you are the author of your pull request:

- Read and understand every line you submit, including the tests and the
  pull request description. Don't submit code you couldn't explain.
- Check that it works. Run the tests yourself, and don't claim results you
  haven't seen.
- Stand by your pull request. Be ready to explain why each change is
  needed, answer reviewers' questions in your own words, and fix what they
  find.
- Make sure you can contribute what you submit under the project's license
  (see the Legal Disclaimer below).

AI-assisted code gets the same review as any other code. A pull request
that its author doesn't seem to have read may be closed.

Pull Requests
-------------

Conin takes contributions through pull requests against ``main``. A pull
request is merged once a maintainer has reviewed it and the CI checks
(formatting, lint, tests, coverage, docs and notebooks) pass.

- Keep changes minimal and focused. Some changes need a longer pull request,
  but aim for one topic per pull request, and put unrelated cleanups in a
  separate one.
- User-visible changes (new features, removals, moved names, changed
  behavior or errors) get an entry in ``CHANGELOG.md`` under the upcoming
  release; moved names also go in its migration table.
- Commit messages: an imperative subject in sentence case, without a final
  period, that says what changed (e.g. "Read gzipped files in load_uai,
  load_cfn and load_json"), and a body that says why when it isn't obvious.
- The pull request template (``.github/pull_request_template.md``) lays
  out the description: it starts with a short paragraph on what was
  wrong or missing and why this approach, then lists, as needed, the API
  changes (new, changed and removed names, new errors), the internal
  changes, and the testing done: the test suite's pass/skip counts, black
  and flake8, the Sphinx builds and the notebooks, and anything that could
  not be tested.
- Before opening a pull request, check the branch with the ``/pr-check``
  Claude Code skill (``.claude/skills/pr-check/``) and paste its summary in
  the template's "Pre-PR check" section, with how you resolved each
  finding. Without Claude Code, go through this file by hand and list what
  you ran. In Claude Code, a project hook
  (``.claude/hooks/pr_check_gate.py``) blocks ``gh pr create`` until the
  skill has checked the commit at HEAD.

Project Conventions
-------------------

The sections below are the one place where the project's conventions are
written down. The package layout, the import layering and the design
decisions are in ``doc/sphinx/architecture.rst``, and the error policy is in
``conin/exceptions.py``.

Online Documentation
--------------------

The user-facing documentation is built from the files in ``doc/sphinx/`` and published
at https://conin.readthedocs.io/en/latest/.

To build it locally from the repository root:

```
python -m pip install -e '.[docs]'
python -m sphinx -b html doc/sphinx doc/sphinx/_build/html
```

Every Python example in the documentation is a Sphinx ``testcode`` or
``doctest`` block, so it is run by the doctest builder (use ``testcode``
rather than ``code-block:: python`` for new examples):

```
python -m sphinx -b doctest doc/sphinx doc/sphinx/_build/doctest
python -m sphinx -W --keep-going -b html doc/sphinx doc/sphinx/_build/html
```

The CI ``docs`` job runs both, with warnings treated as errors. Examples that
need an optional backend declare it with ``:skipif:`` (``not pgmpy_available``,
``not toulbar2_available``, ``not mip_solver_available``; see
``doctest_global_setup`` in ``doc/sphinx/conf.py``), so the docs also build
without the backends; the CI job installs them and sets
``CONIN_DOCS_REQUIRE_BACKENDS=1``, so there every example runs. No example
may need torch: torch is mocked for the API pages.

- Update the docs when a change affects public behavior, the public API,
  examples, optional dependencies, installation or backend requirements.
- Docstrings are NumPy style (``Parameters``, ``Returns``, ``Raises``
  sections) and describe the current behavior. Their ``>>>`` examples run
  only under Sphinx, so they need their own imports, and a ``.. doctest::``
  block with ``:skipif:`` if they need a backend.
- A public function decorated with ``@ovld`` has no docstring for autodoc.
  Make the public function a plain function with the docstring that calls a
  private ovld.
- Remove the generated ``doc/sphinx/_build`` output before committing.

Code Organisation
-----------------

Read ``doc/sphinx/architecture.rst`` (the "Architecture and Design Notes"
page of the documentation) before adding a backend, a constraint family or a
model type. It describes the package layout, the import layering rules
(``graphical_models``, ``util`` and ``exceptions`` import nothing from the
packages above them, and ``pgmpy`` imports only those; ``import conin`` must
not import torch), the representation registry, the argument naming
conventions, immutability and the design decisions behind them.
``conin/tests/test_layering.py`` checks the layering; a new package needs an
entry in its ``LAYERS`` table.

- Public examples live in ``conin/examples/``; tests live next to the code
  they test (``<package>/tests/``).
- ``conin.__all__`` is the top-level public API, and
  ``conin/tests/test_public_api.py`` pins it. Adding or removing a name there
  also means updating ``doc/sphinx/api_conin.rst``.
- Keep public APIs stable when practical. There are no compatibility
  aliases for moved names: when a public name moves, update every importer
  (code, tests, docs, notebooks and ``doc/notebooks/MVR/*.md``) in the same
  change.
- No module shares its name with a function its package re-exports (the
  function would shadow the module).
- Optional dependencies are imported with ``conin.util.try_import``, and
  ``import conin`` must not probe for MIP solvers: library code takes
  ``solver=None`` and calls ``conin.config.mip_solver(solver)``.
- Randomness comes from an ``rng=`` argument (``conin.util.as_rng``), never
  from NumPy's global random state.
- Errors follow the policy in ``conin/exceptions.py``. In particular, raise
  ``InvalidInputError`` for an invalid argument value, not a bare
  ``ValueError``, and ``TypeError`` for an option a function does not take
  (``map_query`` methods reject unknown options).

Code Style
----------

- Code is formatted with ``black`` and must be ``flake8``-clean (settings in
  ``.flake8``); CI fails on any flake8 warning. Complexity (C901) is only
  reported.
- New code reads like the code around it: the same naming, idioms and
  comment density. Comments explain why. Comments and docstrings describe
  the code as it is; the history of a change belongs in the commit message
  and ``CHANGELOG.md``.
- No dead code, unreachable branches, unused options or commented-out code.
  Reuse an existing helper rather than adding a near-copy.

Tests
-----

Submitted code that addresses an issue should include a test exercising the
relevant case. New functionality should include tests to establish validity of
its results and/or effects.

- A regression test should fail without the fix. Tests assert something
  meaningful; don't add tests without assertions or tests that duplicate
  others.
- Tests build small real conin models, and write files only under pytest's
  ``tmp_path``: the suite runs in parallel with pytest-xdist, and tests that
  share files in the source tree delete each other's files.
- Optional dependencies make tests skip when they are missing. A test module
  decides that with the markers of ``conin/tests/optional.py``
  (``needs_pgmpy``, ``needs_torch``), which do not import the package, and
  imports pgmpy where a test uses it: pytest-xdist imports every test module
  in every worker, and pgmpy takes seconds to import.
- CI runs the suite with only HiGHS, only GLPK, and no MIP solver (the
  ``mip-solver-tests`` job), and has no Gurobi license. Tests must not
  hardcode a MIP solver: parametrize over the installed ones, or skip when
  ``conin.config.default_mip_solver is None``.
- Random tests pass a seeded generator as ``rng=``; ``np.random.seed`` has no
  effect on conin's samplers.
- MAP results include a solve time, so compare ``results.solutions``, not
  whole results.
- A new inference method or constraint family is added to the tables of
  ``conin/algorithms/tests/test_random_cross_method.py``, which compares every
  method with brute force. After changing inference, run a larger sweep:
  ``CONIN_RANDOM_CASES=2000 python -m pytest conin/algorithms/tests -n 4``.
- To measure coverage of a package, pass ``--cov`` a path
  (``--cov=conin/constraints/mvr``), not a dotted module name, which imports
  the package too early and breaks the MVR tests.

Run the tests of the code you change, then the whole suite
(``python -m pytest conin -n 4`` with pytest-xdist).

Notebooks
---------

Jupyter notebooks in ``doc/notebooks/`` are committed without outputs. A git
filter handles stripping automatically, but it must be registered once after
cloning:

```
nbstripout --install --attributes .gitattributes
```

``nbstripout`` is included in the dev environment (``dev_environment.yml``).
If you forget this step, CI will reject the pull request with a clear message.
CI also runs every notebook:
``pytest --nbmake --nbmake-kernel=python3 doc/notebooks/*.ipynb doc/notebooks/MVR/*.ipynb``.

Legal Disclaimer
----------------

By contributing to this software project, you are agreeing to the
following terms and conditions for your contributions:

1. You agree your contributions are submitted under the BSD license.
2. You represent you are authorized to make the contributions and grant
   the license. If your employer has rights to intellectual property that
   includes your contributions, you represent that you have received
   permission to make contributions and grant the required license on
   behalf of that employer.
