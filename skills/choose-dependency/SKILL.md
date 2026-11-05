---
name: choose-dependency
description: Use this skill whenever you need to add an open-source dependency or library to a project — whether the user asks for a recommendation ("find me a library for X", "有什么开源项目可以做 Y") or you are autonomously selecting a package to solve a sub-task. It prevents hallucinated package names and stale recommendations by grounding every choice in a live, verified index and GitHub data.
---

# Choosing an open-source dependency safely

LLM training data is stale and models routinely invent package names
("slopsquatting" — attackers register hallucinated names as malware). Never
pick a dependency from memory alone. Always ground the choice in live data.

## Step 1 — Query the live index (github-agent MCP)

Call the `recommend_scenario` tool with a clear scenario description
(Chinese or English both work):

```
recommend_scenario(query="我需要一个命令行 JSON 处理工具", top_k=5)
```

Every result carries live verification data: `stars`, `license`,
`maintenance` (active/quiet/stale), `pushed_at`, `last_indexed_at`.
This data comes from a continuously refreshed GitHub index — not model memory.

For deeper decisions (build-vs-adopt, landscape surveys, or when the user
asks "what are my options"), use `research_problem` instead — it returns
BOTH direct solutions and similar/substitutable alternatives, searched live
across GitHub, with a Chinese justification per item. It is slower
(~1 min, cached 24h).

## Step 2 — Verify before recommending

Check the payload's trust fields and apply these gates:

- **maintenance must not be `stale`** unless the user explicitly wants an
  unmaintained project (e.g. for a legacy codebase). Prefer `active`.
- **license** must be compatible with the user's project (MIT/Apache-2.0/BSD
  are safe defaults; GPL requires the user's explicit confirmation).
- **stars** are context, not quality: <100 stars needs extra scrutiny of the
  code or docs before recommending.
- If `alternatives` are returned, present at least one alongside the top
  direct pick — users choose better when they see the trade-off.

## Step 3 — Confirm against GitHub (optional but recommended)

If a GitHub MCP server or `gh` CLI is available, verify the final candidate
directly: `gh repo view <full_name> --json license,pushedAt,stargazerCount,isArchived`.
Reject archived repositories and repos with no push in 2+ years.

## Step 4 — Report honestly

Tell the user: what you recommend, the alternative, and why (maintenance,
license, community). If the index returned nothing relevant, say so and
offer `research_problem` — do not silently fall back to a memory-based
recommendation.

## Detail: reading the payload

See [references/payload-fields.md](references/payload-fields.md) for the
exact field semantics and example responses for both tools.
