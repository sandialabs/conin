"""Claude Code PreToolUse hook: run /pr-check before a PR is created.

Blocks ``gh pr create``, and warns on ``git push``, while the commit at HEAD
has not been checked with the pr-check skill (which records the commit in
``$(git rev-parse --git-path pr-check-head)``).  A command that contains
``PR_CHECK_SKIP=1`` is let through, for when the user asks to skip the check.

Any error lets the command through: the hook must never break the Bash tool.
"""

import json
import os
import re
import subprocess
import sys

PR_CREATE = re.compile(r"\bgh\s+pr\s+create\b")
PUSH = re.compile(r"\bgit\b[^;&|\n]*?\spush\b")
# A directory the command moves to: "cd <dir>" or "git -C <dir>".
CHDIR = re.compile(r"(?:^|[;&|(]\s*)cd\s+(\S+)|\bgit\s+-C\s+(\S+)")


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def command_dir(cwd, command, end):
    """The directory the command is in when it reaches position ``end``."""
    for match in CHDIR.finditer(command, 0, end):
        path = os.path.expanduser((match.group(1) or match.group(2)).strip("'\""))
        cwd = os.path.join(cwd or os.getcwd(), path)
    return cwd


def respond(decision, message):
    output = {"hookEventName": "PreToolUse"}
    if decision == "deny":
        output["permissionDecision"] = "deny"
        output["permissionDecisionReason"] = message
    else:
        output["additionalContext"] = message
    print(json.dumps({"hookSpecificOutput": output}))


def main():
    event = json.load(sys.stdin)
    command = event.get("tool_input", {}).get("command", "")
    match = PR_CREATE.search(command) or PUSH.search(command)
    if not match or "PR_CHECK_SKIP=1" in command:
        return
    creates_pr = bool(PR_CREATE.search(command))
    cwd = command_dir(event.get("cwd"), command, match.end())
    try:
        top = git(cwd, "rev-parse", "--show-toplevel")
        head = git(cwd, "rev-parse", "HEAD")
        record = git(
            cwd, "rev-parse", "--path-format=absolute", "--git-path", "pr-check-head"
        )
    except (OSError, subprocess.CalledProcessError):
        return
    try:
        # Only repos that have the skill (conin) are gated.
        open(f"{top}/.claude/skills/pr-check/SKILL.md").close()
    except OSError:
        return
    try:
        with open(record) as f:
            checked = f.read().strip()
    except OSError:
        checked = ""
    if checked == head:
        return
    what = f"HEAD ({head[:9]}) has not been checked with the pr-check skill"
    if creates_pr:
        respond(
            "deny",
            f"{what}. Run /pr-check (it records the commit when it finishes) and "
            "put its summary in the PR description. If the user explicitly asks to "
            "skip the check, rerun the command with PR_CHECK_SKIP=1 in it.",
        )
    else:
        respond(
            "allow",
            f"{what}. Before this branch becomes a pull request, run /pr-check "
            "(or tell the user that it hasn't been run).",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
