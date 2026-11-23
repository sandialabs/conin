---
name: pr-check
description: Check a conin branch before it is submitted as a pull request. Runs the CI checks that apply to the change (black, flake8, tests, Sphinx, notebooks), reviews the diff for correctness and code quality, and checks it against the project conventions in CONTRIBUTING.md, doc/sphinx/architecture.rst and conin/exceptions.py (error policy, import layering, no compatibility aliases, immutability, tests, docs, CHANGELOG, commit and PR description style). Use when asked to check, review or vet a branch or PR before submitting it, or "is this ready for a PR?".
---

# Pre-PR check for conin

Check the current branch (or the branch or PR the user names) as a careful
conin maintainer would before merging it. Report findings. Don't edit files
unless the user asks for fixes.

The conventions are written down in the repo, not in this skill. Read them
on the branch being checked before reviewing:

- `CONTRIBUTING.md`: docs, code organisation, code style, tests, notebooks,
  pull requests (CHANGELOG, commit messages, PR descriptions), and the
  commands CI runs.
- `doc/sphinx/architecture.rst`: package layout, import layering,
  representations and dispatch, argument names, immutability, and the
  design decisions (don't undo one without a reason).
- `conin/exceptions.py`: the error policy.

If the PR changes one of these files, check the code against the new
version, and check that the change to the convention is intended.

## 1. Find the change

- In this repo the `nbstripout` git filter may point to a path that doesn't
  exist on this machine. If `git status` or `git diff` fail, add
  `-c filter.nbstripout.clean=cat -c filter.nbstripout.smudge=cat` (and
  `-c diff.ipynb.textconv=cat` for diffs of notebooks).
- Base: the one the user names, else `origin/main`. A branch may be
  stacked on another. List the local branches that HEAD contains but
  `origin/main` doesn't:
  `git for-each-ref --merged HEAD --no-merged origin/main --sort=-committerdate refs/heads --format='%(refname:short)'`.
  If any besides the current branch appear, the nearest one is probably the
  base. Say which base you used. If `git fetch` fails, compare against the
  last-fetched refs and say so.
- Read `git log --format='%h %s%n%b' <base>..HEAD`, `git diff --stat <base>...HEAD`
  and the full diff. Note any uncommitted or untracked changes: they won't
  be in the PR.
- If an earlier run recorded a commit (step 6) that HEAD contains, and
  the user asks for a re-check, review only `<recorded>..HEAD`, plus the
  findings of the earlier run, and rerun the checks those commits affect.
  Say that it was a re-check, and of what.
- Work out what kind of change it is (bug fix, new feature, API change or
  move, refactor, tests only, docs only). The checks below depend on it.

## 2. Run the checks that apply

The commands are in `CONTRIBUTING.md` and `.github/workflows/`. Use the
project's development environment. Workspace notes such as a top-level
`AGENTS.md` may name its location and give faster commands. Run long jobs in
the background. Never `git stash`, switch branches or rewrite the working
tree while a test job is reading it. To run old code (e.g. the base),
extract it with `git archive <base> conin | tar -x -C <scratch>` and put
`<scratch>` on `PYTHONPATH`.

Always:

- black and flake8, as CI runs them. Report C901 complexity only for the
  functions the PR added or changed.
- The tests of every package the diff touches, then the full suite. Always
  run `conin/tests/test_layering.py` and `conin/tests/test_public_api.py`.
- For a bug fix, run the new test against a `git archive` of the base and
  check that it fails there. A regression test that passes on the base
  doesn't test the fix.

When they apply:

- MIP solver code or tests: run the touched tests with only HiGHS, only
  GLPK, and no MIP solver (as the `mip-solver-tests` job does), or at least
  read them to check that they don't assume a solver.
- Inference, scoring or a constraint family changed: the larger
  `CONIN_RANDOM_CASES` sweep.
- `doc/sphinx/`, docstrings with `>>>` examples, or a public name changed:
  both Sphinx builds, with `CONIN_DOCS_REQUIRE_BACKENDS=1` if every backend
  is installed. Build outside the source tree, or delete `doc/sphinx/_build`
  afterwards.
- Notebooks changed, or an API they use changed: the affected notebooks
  with nbmake, and `nbstripout --verify` on the committed notebooks.
- New or changed code paths: coverage of the touched packages, to check that
  the tests reach them.

Record the exact commands and the results (pass/fail/skip counts). If a
check can't run here (no solver, no network, no backend), say so, and don't
report it as passed.

## 3. Review the diff

Read every changed hunk, plus enough of the surrounding code to understand
it. For a large diff, you may split the reading among subagents by package.
Point each one at the convention docs above and give it the diff of its
files. Check its findings yourself before reporting them.

Correctness:

- Logic errors, wrong edge cases (empty inputs, `T=1`, single-state nodes,
  zero probabilities, no feasible solution, `start`/`stop` bounds),
  off-by-one in time indices, label vs internal index mix-ups.
- Every caller of a changed function still works. Grep for the old names
  and signatures across `conin/`, `doc/`, `doc/notebooks/` (including the
  `.md` files) and `README.md`. A moved or renamed name must not leave an
  alias behind.
- Behavior that differs between methods or backends that should agree.
  The cross-method tests compare them with brute force.
- Errors raised have the type the error policy gives, and a message that
  tells the user what to do.
- Code that mutates an argument, a model or a constraint after it is built.

Conventions: go through every section of `CONTRIBUTING.md` and
`architecture.rst` that the change touches, and check each rule against the
diff. Focus matters too, but as a judgment call: a long PR is fine when its
parts belong together. Flag unrelated cleanups mixed into a feature or fix
as "should fix", not as blocking.

## 4. Check the commits and the PR description

- The commit messages follow the style in `CONTRIBUTING.md`.
- `CHANGELOG.md` has an entry for every user-visible change.
- If the user has a PR description (ask, or look for a draft), check that
  it matches the diff, including the test numbers. If there is none, offer
  to draft one in the style in `CONTRIBUTING.md`.

## 5. Report

Lead with a one-line verdict: ready, ready after small fixes, or not ready.
Then:

1. **Checks run**: command, result, and what was skipped and why.
2. **Blocking**: correctness bugs, CI failures, and convention violations
   that reviewers will reject (a fix without a test, a compatibility alias,
   a layering break, torch imported by `import conin`, outputs in
   notebooks).
3. **Should fix**: convention and quality issues that aren't blocking.
4. **Nits**: optional.

Give each finding as `path:line`, the problem, why it matters (cite the
section of the doc), and the fix. Keep the lists short: verify each finding
by reading the code or running it. Leave out anything you can't back up, or
mark it as a question. Don't list things that are fine.

End the report with two blocks the author can paste into the PR template
(`.github/pull_request_template.md`), each in a fenced `markdown` block:

- **Testing**: the template's Testing bullets, filled in from the checks
  you ran (counts, clean or not, what wasn't run and why).
- **Pre-PR check**: one line with the date, the commit checked (short SHA),
  the base and the verdict, then one bullet per blocking or should-fix
  finding, with `Open` after it. The author replaces `Open` with how they
  resolved it. If there were no findings, say so.

If the user has no PR description yet, also offer a full draft that follows
the template.

## 6. Record the run

After reporting, record the commit you checked:

```
git rev-parse HEAD > "$(git rev-parse --git-path pr-check-head)"
```

The project's Claude Code hook (`.claude/hooks/pr_check_gate.py`) uses it
to block `gh pr create`, and to warn on `git push`, while HEAD hasn't been
checked. Record it whatever the verdict was. The hook only requires that the
check ran, and the author decides what to fix. If there are uncommitted
changes, say that the record covers only HEAD.
