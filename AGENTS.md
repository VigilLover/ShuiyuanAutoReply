# Repository Guidelines

## Project Structure & Module Organization

Python code lives under `src/shuiyuan_auto_reply/` (setuptools src layout). Ports-and-adapters:

- `domain/` — core types only
- `application/` — orchestration, handlers, ports (`BotService` is shared by forum worker and HTTP API)
- `infrastructure/` — integrations (forum, LLM, persistence, prompts, retrieval, tools)
- `interfaces/` — CLI, worker, FastAPI entry points
- `bootstrap/` — composition root / settings wiring
- `features/`, `prompts/` — feature modules and packaged prompt text

Vue 3/TypeScript frontend is in `web/src/`; **built assets are committed** under `src/shuiyuan_auto_reply/interfaces/api/static/` and ship in the wheel. Do not hand-edit those generated files.

Operational helpers live in `scripts/` (CI checks, deploy). Compose manifests are in `deploy/`. Deployment config samples are in `config/`.

## Commands

```bash
# Install (prefer uv; CI uses --locked)
uv sync --extra dev --extra server
# Local embedding / Neo4j full stack also needs:
uv sync --extra dev --extra server --extra local-embedding --extra neo4j

# Run
uv run shuiyuan-bot                      # forum worker (default persona wolf_lumine)
uv run shuiyuan-bot wolf_lumine --web    # worker + management UI (needs [server])
uv run shuiyuan-api                      # HTTP API / web console only
uv run shuiyuan-ops config check         # CLI config validation

# Verify (order matches CI)
uv run black --check src test scripts deploy/mcp
uv run isort --check-only src test scripts deploy/mcp
uv run pytest -q                         # offline; --run-live for external services
npm --prefix web test
npm --prefix web run build               # vue-tsc --noEmit && vite build
```

`pytest -q` is the default suite. Tests marked `@pytest.mark.live` are skipped unless `--run-live` is passed. Run a single file/test as usual: `uv run pytest test/test_forum_auth.py -q`.

Frontend rebuild after `web/` changes:

```bash
npm --prefix web install
npm --prefix web run build
```

Then commit the regenerated `src/shuiyuan_auto_reply/interfaces/api/static/` output with the source change.

## Architecture Boundaries (enforced by tests)

`test/test_architecture_boundaries.py` fails the suite if you break these:

- `application/` must not import `aiohttp`, `fastapi`, `sqlalchemy`, `neo4j`, `neomodel`, or `infrastructure`
- `domain/` stays framework-free (only stdlib `dataclasses`/`enum` plus domain modules)
- Production code must not import `examples`

When changing ports, providers, or dependency direction, extend `test_port_contracts.py` / architecture tests rather than only fixing the import.

## Frontend & Packaging

- `npm --prefix web run build` type-checks and writes to `interfaces/api/static/`
- Prompt package data (`prompts/*.json`, `personas/`, `policies/`, `capabilities/`) is listed in `pyproject.toml` `[tool.setuptools.package-data]`; CI extracts a wheel and runs `scripts/ci/check_prompt_package.py`. Adding prompt files requires updating both the files and package-data.
- Managed prompts use `prompts/manifest.json` (current rule version) plus `prompts/legacy_defaults.json` for migration hashes. See `docs/agent-tools.md` for prompt/profile behavior.

## Config, State, Secrets

- Copy `.env.example` → `.env` for local runs. Environment defaults are documented there.
- State dir defaults to `~/.shuiyuan-auto-reply` (`SHUIYUAN_STATE_DIR` overrides): `state.sqlite3`, `master.key` (0600), `artifacts/`.
- Never commit: API keys, cookies (`cookies`), `secrets/`, state-dir contents, or real `config/deployment.toml` values. `config/deployment.example.toml` is the safe template.
- Management UI binds `127.0.0.1:11451` by default; do not assume auth exists if exposed.
- Forum and web runtimes use isolated Prompt / Session / long-term-memory namespaces; web/forum agents pin DeepSeek vision (`deepseek-v4-flash-vision-exp` + `DEEPSEEK_API_KEY`) with no fallback.

## Testing Quirks

- Default suite is offline (mocked HTTP, in-memory SQLite via `tmp_path` / `SHUIYUAN_STATE_DIR`).
- Live tests: model APIs, real forum, MCP — never run them as part of routine verification.
- After any change that could affect agent loops/tool pairing, prefer the dedicated regressions (`test_tool_call_pairing.py`, `test_convergence_regressions.py`, fixtures under `test/fixtures/agent_convergence/`).

## Branch / Release

- `main` is the release source and default branch; `dev` is the integration branch (PRs into main). CI runs on PRs and pushes to `dev`/`main`/`remote-deploy`.
- Formal releases tag `vX.Y.Z` from `main` only. `deploy/release-policy.json` currently `backward-compatible` / `recent:2` — schema/migration changes must stay compatible with recent releases or update that policy deliberately.
- Do not invent deploy steps; follow `docs/cicd.md`, `docs/first-deployment.md`, `docs/branch-strategy.md`.

## Style

- Python: four-space indent, Black-compatible; isort profile `black`. Sort/format before submit (`black src test scripts deploy/mcp` + matching isort). Type hints on public boundaries.
- Vue components `PascalCase.vue`; TS: two spaces, single quotes, no semicolons (see `web/`).
- Commits: concise Conventional Commit subjects (`feat:`, `fix:`, `test:`). Keep commits scoped.

## High-value docs

- `docs/agent-tools.md` — agent tools, media limits, prompt management (read before touching tool schemas or prompt composition)
- `docs/deployment.md` — local vs remote profiles, re-vectorization
- `docs/cicd.md` — CI/release/deploy contract
- `README.md` — run modes and HTTP compatibility endpoints
