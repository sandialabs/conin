<!--
See "Pull Requests" in CONTRIBUTING.md. Delete sections that don't apply.
-->

What was wrong or missing, and why this approach.

**API**

- New, changed and removed public names, and new errors. Moved names have
  no compatibility aliases; every importer is updated.

**Internals**

-

**Testing**

- Full suite: N passed, N skipped.
- `black --check .` and `flake8 .` are clean.
- Sphinx doctest and HTML builds (if docs or docstrings changed).
- Notebooks (if they or an API they use changed).
- Not tested, and why:

**Pre-PR check**

<!--
Paste the summary that the /pr-check skill prints at the end, starting with
"Checked `<short SHA>` against `<base>` on <date>: <verdict>." for the
commit at the branch tip, and how each finding was resolved. Without Claude
Code, go through CONTRIBUTING.md by hand and list what you ran instead.
-->

- [ ] `CHANGELOG.md` has an entry for every user-visible change.
- [ ] A bug fix comes with a test that fails without it.
