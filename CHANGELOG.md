# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-11-04

### Added
- **MCP server** (`POST /mcp`): stateless Model Context Protocol endpoint exposing
  `recommend_scenario` / `list_projects` / `get_project` tools so AI coding agents
  can query a live, verified repository index; optional `MCP_AUTH_TOKEN` bearer auth.
- **Trust signals**: `license` column (auto-migrated), `pushed_at` and a `maintenance`
  tier (active/quiet/stale) on every project payload; long-unmaintained repos decay
  slightly in recommendation ranking, unknown push time is never punished.
- **Weekly digest** `GET /api/admin/digest?days=7`: top/zero-result queries, voted
  projects and newly indexed repos — the input for keyword expansion and content.
- Install/deploy/release scripts: `setup_windows.bat` (one-command install with
  randomized secrets), `deploy_local.bat` (pull → sync → test gate → restart),
  `restart_api.bat`; tagged GitHub Releases via `.github/workflows/release.yml`;
  pinned known-good environment in `requirements.lock`.

### Fixed
- Evolution promotion deadlocked the whole service (non-reentrant `MODEL_SWAP_LOCK`
  re-acquired during re-embed); promotion and release-pull now re-embed outside the lock.
- Queue starvation: `analysis_attempts` cap + attempts-first ordering so permanently
  failing projects no longer occupy the analysis queue head forever.
- Cross-process races: file-lock mutex for fetch/refresh/patrol/evolution jobs across
  API + worker processes; per-row savepoint upsert instead of all-or-nothing batch.
- Vector space consistency: Chroma records the embedder fingerprint, wipes on mismatch,
  and re-embed/rollback paths park stale vectors for patrol instead of mixing spaces.
- Voice routes blocked the event loop (whisper/TTS now run in the threadpool); upload
  size cap enforced (`max_upload_mb` was dead config); text length bounds on inputs.
- Rate limiting behind the tunnel: optional trusted proxy headers (`TRUST_PROXY_HEADERS`,
  loopback-only) plus a tracked-client cap; refresh aborts on rate limit instead of
  sleeping for hours.
- Security: `torch.load(weights_only=True)`, tar extraction `filter="data"`, JWT
  requires numeric `exp`, startup warning on default secrets, training subprocess
  timeout, atomic evolution history writes, deploy workflow fails on missing
  `DEPLOY_ADMIN_TOKEN` and generates a fresh `SECRET_KEY` per deploy.
- TF-IDF fallback was near-random for Chinese queries (no tokenizer): CJK bigrams.
- Repo hygiene: stopped tracking `github_agent.db-shm/-wal` and training data.

## [0.1.0] - 2026-09-06

Initial public release: scheduled GitHub retrieval, pluggable LLM analysis with a
self-trained local model, semantic scenario recommendations, offline voice, GitHub
OAuth, 24/7 patrol/self-evolution loop, Cloudflare tunnel deployment.
