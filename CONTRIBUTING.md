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

When documentation examples change, also run the Sphinx doctest builder:

```
python -m sphinx -b doctest doc/sphinx doc/sphinx/_build/doctest
```

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
