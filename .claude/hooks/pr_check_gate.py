"""Claude Code PreToolUse hook: run /pr-check before a PR is created.

Blocks ``gh pr create`` unless the PR description's "Pre-PR check" section
names the commit at HEAD (the pr-check skill writes the line
``Checked `<short SHA>` against `<base>` on <date>: <verdict>.``).  The
description is read from the command itself (``--body``, or a heredoc
piped to ``--body-file -``) and from the file given to ``--body-file``/``-F``.
A command that contains ``PR_CHECK_SKIP=1`` is let through, for when the user
asks to skip the check.

Any error lets the command through: the hook must never break the Bash tool.
"""

import json
import os
import re
import subprocess
import sys

# Only at the start of a command (after optional VAR=value assignments), so
# that the words inside a commit message or a search pattern don't match.
START = r"(?:^|[;&|(\n])\s*(?:\w+=\S*\s+)*"
PR_CREATE = re.compile(START + r"gh\s+pr\s+create\b")
# A directory the command moves to: "cd <dir>" or "git -C <dir>".
CHDIR = re.compile(r"(?:^|[;&|(]\s*)cd\s+(\S+)|\bgit\s+-C\s+(\S+)")
BODY_FILE = re.compile(r"(?:--body-file[=\s]|-F\s*)(\S+)")
SHA = re.compile(r"\b[0-9a-f]{7,40}\b")


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


def description(command, cwd):
    """The command text plus the contents of the files it passes as the body."""
    parts = [command]
    for match in BODY_FILE.finditer(command):
        path = match.group(1).strip("'\"")
        if path == "-":
            continue
        try:
            with open(os.path.join(cwd, os.path.expanduser(path))) as f:
                parts.append(f.read())
        except OSError:
            pass
    return "\n".join(parts)


def names_commit(text, head):
    """Whether a "Pre-PR check" section of ``text`` names a prefix of ``head``."""
    for section in text.split("Pre-PR check")[1:]:
        if any(head.startswith(sha) for sha in SHA.findall(section)):
            return True
    return False


def deny(message):
    output = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": message,
    }
    print(json.dumps({"hookSpecificOutput": output}))


def main():
    event = json.load(sys.stdin)
    command = event.get("tool_input", {}).get("command", "")
    match = PR_CREATE.search(command)
    if not match or "PR_CHECK_SKIP=1" in command:
        return
    cwd = command_dir(event.get("cwd"), command, match.end())
    try:
        top = git(cwd, "rev-parse", "--show-toplevel")
        head = git(cwd, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return
    # Only repos that have the skill (conin) are gated.
    if not os.path.isfile(f"{top}/.claude/skills/pr-check/SKILL.md"):
        return
    if names_commit(description(command, cwd), head):
        return
    deny(
        f"The PR description's Pre-PR check section doesn't name HEAD "
        f"({head[:9]}). Run /pr-check and put its summary, starting with "
        '"Checked `<short SHA>` against `<base>` on <date>: <verdict>.", in '
        "the description. If the user explicitly asks to skip the check, rerun "
        "the command with PR_CHECK_SKIP=1 in it."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
