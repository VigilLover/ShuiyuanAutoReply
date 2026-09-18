# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` at the repo root is the maintained contributor guide (structure, style, branch/release rules, secrets). Read it too; this file focuses on commands and on the cross-file architecture that is not obvious from any single module.

## Commands

Python is managed with `uv` (CI pins `uv sync --locked`; never regenerate `uv.lock` casually). Frontend is Vue 3 + Vite in `web/` and needs Node ≥ 22.18.

```bash
uv sync --extra dev --extra server                      # minimal (matches CI "remote" matrix)
uv sync --extra dev --extra server --extra local-embedding --extra neo4j   # full local stack

uv run shuiyuan-bot                                     # forum worker, persona wolf_lumine
uv run shuiyuan-bot wolf_lumine --web                   # worker + management UI on 127.0.0.1:11451
uv run shuiyuan-api                                     # management UI / web runtime only
uv run shuiyuan-ops config check                        # validate config (also: doctor, db migrate, corpus/memory export|import, cookie convert, backup, restore)
uv run shuiyuan-ops --config config/deployment.toml --profile remote config check   # --config/--profile go BEFORE the subcommand

# Verification, same order as CI
uv run black --check src test scripts deploy/mcp
uv run isort --check-only src test scripts deploy/mcp
uv run pytest -q                                        # offline suite; live tests auto-skipped
uv run pytest -q test/test_forum_auth.py                # single file
uv run pytest -q test/test_tool_call_pairing.py -k missing_result   # single test
uv run pytest -q --run-live                             # ONLY when explicitly asked: real forum/model/MCP calls
npm --prefix web test                                   # node --test over web/test/*.test.ts
npm --prefix web run build                              # vue-tsc + vite; writes interfaces/api/static/ (committed, never hand-edit)
```

Format before committing: `uv run black src test scripts deploy/mcp && uv run isort src test scripts deploy/mcp`.

Tests that touch state must isolate it with `tmp_path` / `SHUIYUAN_STATE_DIR`; the default state dir is `~/.shuiyuan-auto-reply`.

## Architecture

### Two channels, one application core

Everything funnels through `application/bot_service.py:BotService`. A channel builds a `domain.ReplyRequest`, `BotService` acquires a slot from the shared `application/scheduling.py:ReplyScheduler` (concurrency 3, queue 100, 900 s timeout, read from `[common.runtime]`), loads history from a `SessionRepository`, walks the `HandlerRegistry` in priority order, and persists the `ReplyResult`. Handlers are the `MessageHandler` protocol in `application/dispatch.py`; concrete ones live in `application/handlers/` (help, pet/`【rua】`, clear, dice, poll, chat). `ChatHandler` always matches and is last; it delegates to a `ChatBackend`, in practice `infrastructure/llm/legacy_chat.py:LegacyMentionChatBackend` wrapping a `MentionChatModel`.

- **Forum channel** (`interfaces/worker/main.py:run_worker`): creates `shuiyuan/shuiyuan_model.py:ShuiyuanModel` (cookie-authenticated Discourse client), then `features/mention/mention_model.py:MentionModel`, which subclasses `shuiyuan/user_action_model.py:BaseUserActionModel`. `watch_new_action_routine` polls notifications, enqueues into `infrastructure/persistence/work_queue.py:ForumQueue` (table `forum_jobs` in the same `state.sqlite3`), and runs `_prepare_action → _accept_action → _execute_action`. Trigger tokens (`【小狼】`, `【帮助】`, `【rua】`, `【清除历史】`, dice/poll) are matched in `MentionModel`'s condition methods and then handed to `BotService.match` / `reply_matched`. Admission is keyed per **post**, not per topic. Status flow and `needs_review` semantics are in `docs/forum-monitor.md`.
- **Web channel** (`interfaces/api/app.py:create_app` + `bootstrap/container.py:ApplicationContainer.for_api`): uses `LazyForum`/`LazyChat` so the UI starts even without a forum login or API key. Handlers are help, `【rua】`, chat. The service is wrapped in `_SwappableBotService` so the web runtime can be hot-swapped from the settings page.

Boundary rules (enforced by `test/test_architecture_boundaries.py`): `application/` never imports `infrastructure/`, aiohttp, fastapi, sqlalchemy, neo4j; `domain/` is stdlib-only. `application/ports/` holds the Protocols infrastructure implements; extend `test/test_port_contracts.py` when adding or changing one.

### Provider selection and the agent graph

`bootstrap/providers.py:MentionProviderFactory` is the only place that builds the chat model. DeepSeek is the sole chat provider (`features/mention/mention_deepseek_model.py:MentionDeepSeekModel`, either Responses API or chat completions per `DEEPSEEK_MENTION_API_FORMAT`). It subclasses `features/mention/mention_chat_model.py:MentionChatModel`, which builds a LangGraph `StateGraph`:

```
retrieve_style_context → load_topic_context → load_long_term_memory → [load_current_images → load_replied_post_images]
→ prepare_messages → call_model ⇄ (log_tool_calls → validate_tool_calls → tools → log_tool_outputs → [collect_tool_output_images])
→ finalize_response → save_history
```

Node bodies are grouped in `features/mention/chat_pipeline.py`. `features/mention/context_budget.py` does token-budget compaction and **tool call/result pairing repair**; breaking pairing yields a 400 from OpenAI-compatible Responses endpoints, so run `test_tool_call_pairing.py` and `test_convergence_regressions.py` after touching the loop. Forum/web/user tools are exposed through `features/mention/shuiyuan_tools_wrapper.py` and `tool_catalog.py` (one tool per capability: `forum_search`, `forum_read`, `users`, `web_search`, `web_read`; legacy names are remapped by `migrate_tool_names`); `web_*` come from the external SimpleMCP server via `langchain-mcp-adapters`. Details and limits: `docs/agent-tools.md`.

### Configuration layering

Two systems coexist:

- `bootstrap/deployment.py:load_deployment(config, profile)` reads `config/deployment.toml` (`[common.*]` plus `[profiles.local.*]` / `[profiles.remote.*]`) into a process-global reachable via `get_deployment()`. Precedence: CLI args > profile > common > legacy `.env` > defaults. Secret fields accept `{env="NAME"}` or `{file="path"}`. Without `--config`, only `.env` is used and the profile is `local`.
- `bootstrap/settings.py` dataclasses (`AppSettings`, `ProviderSettings`, …) read environment variables directly; `.env.example` documents them. `constants.py:settings` is a legacy accessor that now delegates to the deployment config for embeddings.

`local` profile means m3e-base embeddings + Neo4j retrieval; `remote` means OpenAI-compatible embeddings + PostgreSQL/pgvector, verified at worker start by `check_vector_space`. Retrieval adapters are in `infrastructure/retrieval/`, embeddings in `infrastructure/embedding/`.

### Runtime profiles, prompts, and hot swap

Per-scope (`"forum"` / `"web"`) runtime profiles live in SQLite via `SQLiteStateStore.get_profile` and carry model, base URL, API format, `enabled_tools`, `disabled_mcp_tools`, and prompt settings, with an `active_revision`. The forum worker's `refresh_runtime` closure rebuilds a `MentionChatModel` when the revision changes and calls `MentionModel.swap_chat_model`; the web side goes through `ApplicationContainer.prepare_runtime_profile` / `activate_prepared_runtime`. Stored model configs (`infrastructure/persistence/model_configs.py`) and their keys in the `LocalSecretVault` override profile and env values (`apply_profile_endpoint`).

Managed prompts are composed by `infrastructure/prompts/profiles.py:render_profile` from `prompts/manifest.json` + `prompts/personas/*.txt`, `policies/*.txt`, `capabilities/*.txt`. Adding a prompt file requires adding it to `[tool.setuptools.package-data]` in `pyproject.toml`; CI builds a wheel and runs `scripts/ci/check_prompt_package.py`. Forum and web use separate prompt, session, and long-term-memory namespaces.

### State and observability

`infrastructure/persistence/state.py:SQLiteStateStore` owns `state.sqlite3` (conversations, messages, runs, run_events, artifacts, runtime_profiles, prompt_versions, model_configs, secret_values, …). `application/events.py` sets a contextvar execution context so any code in a reply can `emit_event`; `SQLiteExecutionObserver` persists those into `runs`/`run_events`, which the API streams over SSE (`/api/forum/events/stream`) to the Vue UI. Long-term memory (`langmem`) and style-example retrieval live in Postgres/Neo4j, not SQLite.

### Legacy flat packages

`shuiyuan/`, `database/`, `embeddings.py`, `retry.py`, and `constants.py` predate the ports-and-adapters refactor. They are still the real forum client and Postgres/Neo4j managers used by `features/` and `infrastructure/`; treat them as infrastructure, do not import them from `application/` or `domain/`, and prefer adding new adapters under `infrastructure/` rather than growing them.

## Deployment and release contract

Production is `deploy/Dockerfile` (multi-stage: Node build → uv → slim runtime, non-root, runs `shuiyuan-bot --config /etc/shuiyuan/deployment.toml --profile remote --web`) plus `deploy/compose.yaml` (bot, postgres/pgvector, SimpleMCP). Releases are `vX.Y.Z` tags on `main` only; `.github/workflows/release.yml` re-runs CI, builds three images, runs `scripts/ci/integration.py` against `deploy/compose.test.yaml`, then publishes. `deploy/release-policy.json` (`backward-compatible`, `recent:2`) means schema/persistence changes must stay readable by the two previous releases or the policy must be changed deliberately. Do not invent deploy steps; follow `docs/cicd.md`, `docs/deployment.md`, `docs/first-deployment.md`, `docs/branch-strategy.md`.

Never commit `cookies`, `secrets/`, `.env`, real `config/deployment.toml`, or state-dir contents.
