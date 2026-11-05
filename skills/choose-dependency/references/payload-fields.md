# Payload field semantics (github-agent MCP)

## recommend_scenario → structuredContent

| field | meaning | gate |
| --- | --- | --- |
| `data_source` | always `live-github-index` — proof the data is real, not model memory | must be present |
| `full_name` | GitHub owner/repo | use for `gh repo view` |
| `stars` / `forks` | live counters at index time | context only |
| `license` | SPDX id (`MIT`, `Apache-2.0`, ...), `null` = unknown/unlicense | compatibility gate |
| `maintenance` | `active` (pushed ≤180d) / `quiet` (≤730d) / `stale` (>730d) / `unknown` | reject `stale` without user consent |
| `pushed_at` | last commit push, ISO | secondary to `maintenance` |
| `last_indexed_at` | when this project was last refreshed in the index | older than 14d → prefer re-verification |
| `score` | similarity (or fused hybrid score), normalized 0-1 | ordering hint, not quality |
| `use_cases` | Chinese scenario labels from the analysis pipeline | match against user intent |
| `reason` | LLM-generated justification (only with `RERANK_PROVIDER=llm`) | helpful but verify facts |

## research_problem → structuredContent

Same project fields, split into two groups:

- `results.direct` — projects whose core purpose solves the stated problem
- `results.alternatives` — projects with similar/overlapping functionality
  usable as substitutes (different approach or form factor)
- `summary` — short Chinese synthesis contrasting the two groups
- `keywords` — the generated English search plan (transparency)

Present at least one `alternatives` entry whenever it is non-empty.

## list_projects / get_project

Same project fields; `get_project` adds `found: true|false`.
