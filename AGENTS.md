# AGENTS.md

The project's conventions (code, tests, docs, notebooks, pull requests) are
in `CONTRIBUTING.md`; the package layout, the import layering and the design
decisions are in `doc/sphinx/architecture.rst`; the error policy is in
`conin/exceptions.py`. Read them before changing code, and follow them. This
file adds only what is specific to coding agents.

- Keep changes minimal and focused. Don't revert unrelated user or
  collaborator changes.
- Run the tests of the code you change, and the full suite when feasible.
  Don't report a check as passed if you could not run it.
- Before a branch is submitted, check it with the `pr-check` skill
  (`.claude/skills/pr-check/SKILL.md`), and put its summary in the PR
  description (`.github/pull_request_template.md`). Agents without skill
  support can read `SKILL.md` and follow it.
- When you learn a new project convention, add it to `CONTRIBUTING.md` (or
  `architecture.rst` for a design decision), not here, so it is written down
  in one place.
