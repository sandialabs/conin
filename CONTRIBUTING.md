Contributing to Conin
=====================

Online Documentation
--------------------

The user-facing documentation is built from the files in ``doc/sphinx/`` and published
at https://conin.readthedocs.io/en/latest/.

To build it locally from the repository root:

```
python -m pip install -e .[docs]
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

Code Organisation
-----------------

The package layout, the import layering rules (``graphical_models``,
``common``, ``util`` and ``exceptions`` import nothing from the packages
above them; ``import conin`` must not import torch), the representation
registry and the design decisions behind them are described in
``doc/sphinx/architecture.rst`` (the "Architecture and Design Notes" page of
the documentation). Read it before adding a backend, a constraint family or a
model type.

- Public examples live in ``conin/examples/``; tests live next to the code
  they test (``<package>/tests/``).
- Optional dependencies are imported with ``conin.util.try_import``; tests
  that need them skip when they are missing.
- There are no compatibility aliases for moved names: when a public name
  moves, update every importer (code, tests, docs, notebooks).
- Code is formatted with ``black``.

Pull Requests
-------------

Conin manages source code contributions via pull requests. For a pull request
to be accepted, it must satisfy all code integration and coverage tests.

Submitted code that addresses an issue should include a test exercising the
relevant case. New functionality should include tests to establish validity of
its results and/or effects.

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
