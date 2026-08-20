---
description: Stage, commit, and push all changes to GitHub. Use when user says "commit changes", "push to github", "commit everything", or similar.
---

# Git Push

Commit and push current changes to the remote repository.

## Usage

User says: "commit changes to github", "push changes", "commit everything"

## Procedure

1. **Check status**: `git status` to see staged, unstaged, and untracked files
2. **Check diff**: `git diff` and `git diff --staged` to understand all changes
3. **Check recent commits**: `git log --oneline -5` to match commit message style
4. **Stage all**: `git add -A`
5. **Commit**: Use a concise message matching the repo's style. Focus on *why* not *what*:
   - New feature: "add <feature>"
   - Fix: "fix <issue>"
   - Cleanup: "clean up <area>"
   - Config: "update <config>"
6. **Push**: `git push` (or `git push -u origin <branch>` if no upstream)
7. **Verify**: `git status` to confirm clean state

## Safety

- Never force push (`--force`) unless explicitly requested
- Never skip pre-commit hooks (`--no-verify`) unless explicitly requested
- Never push to main/master without explicit request
- If pre-commit hook fails, fix the issue and create a new commit (don't amend)
