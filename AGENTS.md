# Repository Guidelines

## Project Structure & Module Organization

Python code lives under `src/shuiyuan_auto_reply/`. In this ports-and-adapters layout, `domain/` holds core types, `application/` contains orchestration and ports, `infrastructure/` implements integrations, and `interfaces/` provides CLI, worker, and FastAPI entry points. Feature code is grouped in `features/`; prompt text is in `prompts/`. Tests live in `test/`. The Vue 3/TypeScript frontend is in `web/src/`; its generated build output goes to `src/shuiyuan_auto_reply/interfaces/api/static/`. Operational helpers belong in `scripts/`.

## Build, Test, and Development Commands

- `uv sync --extra dev --extra server` installs Python 3.12 dependencies from `uv.lock`.
- `uv run shuiyuan-bot` starts the forum worker; add `wolf_lumine --web` for the management UI.
- `uv run shuiyuan-api` starts only the HTTP API and web console.
- `uv run pytest -q` runs offline tests; `--run-live` opts into external-service tests.
- `npm --prefix web install` installs frontend dependencies.
- `npm --prefix web run dev` starts Vite locally; `npm --prefix web run build` type-checks and creates production assets.

## Coding Style & Naming Conventions

Use four-space indentation and Black-compatible Python; sort imports with isort (`profile = "black"`). Use `snake_case` for functions/modules, `PascalCase` for classes, and type hints on public boundaries. Keep application and domain code independent of FastAPI, aiohttp, and database SDKs. Vue components use `PascalCase.vue`; TypeScript uses two spaces, single quotes, and no semicolons. Run `black src test` and `isort src test` before submitting Python changes.

## Testing Guidelines

Name pytest files `test_*.py` and tests `test_<behavior>`. Add contract or architecture tests when changing ports, providers, or dependency boundaries. Mark credential- or network-dependent tests with `@pytest.mark.live`; keep the default suite offline. Build the frontend after changes under `web/`.

## Commit & Pull Request Guidelines

Recent history favors concise Conventional Commit subjects such as `feat: add multimodal support` and `fix: tool trim`. Keep commits scoped. Pull requests should explain the change, list verification commands, note configuration impacts, and link issues. Include screenshots for visible UI changes and identify new environment variables.

## Security & Configuration

Copy `.env.example` for local configuration, but never commit API keys, database credentials, cookies, or files from the state directory (`~/.shuiyuan-auto-reply` by default). The management UI binds to `127.0.0.1` by default; add authentication at the proxy before exposing it beyond localhost.
