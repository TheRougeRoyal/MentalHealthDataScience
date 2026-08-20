---
name: repo-review
description: Read and explore a repository to produce a structured overview of its architecture, key files, dependencies, and patterns. Triggered by "read this repository", "read this codebase", or similar inspection requests.
---

# Repository Review

Systematic exploration of an unfamiliar or returning repository to produce a structured overview.

## When to use

- User says "read this repository", "read this codebase", "look at this repo"
- User wants to understand a project before working on it
- Session starts with an unfamiliar codebase

## Procedure

### 1. Identify project root and structure

```
read <project_root>           # directory listing
read <project_root>/README.md  # if exists
```

### 2. Identify the tech stack

Look for:
- **Backend entry point**: `server.js`, `app.py`, `main.py`, `index.js`, `cmd/` (Go)
- **Frontend entry point**: `index.html`, `App.tsx`, `App.vue`, `main.js`
- **Package manifests**: `package.json`, `requirements.txt`, `pyproject.toml`, `Cargo.toml`, `go.mod`
- **Config**: `tsconfig.json`, `vercel.json`, `railway.json`, `Dockerfile`, `docker-compose.yml`
- **Tests**: `tests/`, `__tests__/`, `*.test.*`, `*.spec.*`

### 3. Read key files (in order of importance)

1. `README.md` — project purpose, setup, architecture
2. Package manifest — dependencies, scripts, engine requirements
3. Entry point(s) — server startup, route mounting, middleware
4. Source directory structure — `src/`, `lib/`, `app/`, `api/`
5. Config files — environment, deployment, database
6. Test structure — test framework, coverage, fixtures

### 4. Produce structured overview

Output format:

```
## Project: <name>
**Stack:** <languages, frameworks, databases>
**Purpose:** <one-line description>

### Structure
<directory tree with 1-line annotations>

### Key files
- `<path>` — <purpose>

### Dependencies
<list notable deps and why they're used>

### Patterns observed
<architectural patterns, conventions, notable decisions>
```

### 5. Stop conditions

- Do NOT modify any files (read-only unless user explicitly asks for changes)
- Do NOT create tasks unless user follows up with work
- Stop after producing the overview and await next instruction

## Anti-patterns

- Reading every file exhaustively — focus on the 10-15 most important files
- Producing code diffs or suggestions unprompted — this is an inspection, not a refactor
- Creating tasks or plans without being asked
