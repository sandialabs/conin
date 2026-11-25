"""Record that the pr-check skill has checked the commit at HEAD.

Run as ``python3 <checkout>/.claude/hooks/record_pr_check.py`` at the end of
/pr-check.  It appends HEAD of the checkout this script lives in to
``$(git rev-parse --git-common-dir)/pr-check-head``, one commit per line.  The
record is shared by the main checkout and its worktrees, so a check made in a
worktree counts for ``pr_check_gate.py`` and ``open_pr.command`` too.

The command takes no arguments, so that a user can allow exactly this command
in their Claude Code permissions.
"""

import pathlib
import subprocess
import sys


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def main():
    checkout = pathlib.Path(__file__).resolve().parents[2]
    head = git(checkout, "rev-parse", "HEAD")
    common = git(checkout, "rev-parse", "--path-format=absolute", "--git-common-dir")
    record = pathlib.Path(common) / "pr-check-head"
    with open(record, "a") as f:
        f.write(head + "\n")
    print(f"Recorded {head[:9]} in {record}")


if __name__ == "__main__":
    sys.exit(main())
