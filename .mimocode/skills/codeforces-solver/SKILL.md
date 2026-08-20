---
name: codeforces-solver
description: Solve competitive programming problems from Codeforces. Fetches problem statements, analyzes constraints, and produces optimized C++17 solutions. Triggered by pasting a CF problem text or URL.
---

# Codeforces Problem Solver

Solve competitive programming problems with optimized C++17 solutions.

## When to use

- User pastes a Codeforces problem statement (text or URL)
- User says "solve this problem" with a CF link
- User asks for help with a competitive programming problem

## Procedure

### 1. Parse the problem

- If URL: fetch the problem page with `webfetch`, extract statement, constraints, examples
- If text: parse directly for problem statement, input/output format, constraints, examples

### 2. Analyze

- Identify problem type: DP, graph, greedy, math, data structures, string, etc.
- Note constraints: `n ≤ ?`, time limit, memory limit — these determine algorithm choice
- Work through examples by hand to verify understanding
- Consider edge cases: `n=1`, all-equal values, maximum constraints, negative numbers

### 3. Design algorithm

- State the approach in 1-2 sentences
- Identify time complexity and verify it fits within limits
- For hard problems (*2500+): consider if editorial/hints are needed

### 4. Implement

- Write clean, readable C++17 code
- Use `#include <bits/stdc++.h>` for竞赛编程
- Include `using namespace std;`
- Use `typedef long long ll;` when needed
- Handle I/O efficiently: `ios_base::sync_with_stdio(false); cin.tie(NULL);`
- Add comments for non-obvious logic

### 5. Verify

- Trace through all examples mentally
- Check for off-by-one errors, integer overflow, edge cases
- Verify complexity matches constraints

### 6. Output

Provide:
1. **Approach** — 1-2 sentence algorithm description
2. **Complexity** — Time and space with justification
3. **Complete code** — Ready to copy-paste and submit

## Constraints reference

| Constraint | Likely approach |
|---|---|
| n ≤ 20 | Brute force / bitmask DP |
| n ≤ 1000 | O(n²) DP or graphs |
| n ≤ 10^5 | O(n log n) sorting / segment trees |
| n ≤ 10^6 | O(n) linear scan / two pointers |
| n ≤ 10^9 | Math / binary search / formula |
| Graph n ≤ 2000 | Floyd-Warshall, adjacency matrix |
| Graph n ≤ 10^5 | BFS/DFS, Dijkstra, Union-Find |

## Anti-patterns

- Do NOT use Python — user expects C++17 solutions
- Do NOT skip examples — always verify against them
- Do NOT ignore constraints — they determine the algorithm
- Do NOT output partial solutions — provide complete, submittable code
