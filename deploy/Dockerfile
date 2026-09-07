# syntax=docker/dockerfile:1
FROM node:22.18.0-bookworm-slim AS frontend
WORKDIR /build/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.8.15 AS uv
FROM python:3.12.11-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra server --no-install-project --no-editable
COPY src ./src
COPY README.md ./
COPY --from=frontend /build/src/shuiyuan_auto_reply/interfaces/api/static ./src/shuiyuan_auto_reply/interfaces/api/static
RUN uv sync --frozen --no-dev --extra server --no-editable
RUN .venv/bin/python -c "import importlib.util; assert all(importlib.util.find_spec(m) is None for m in ('torch','sentence_transformers','neo4j','neomodel'))"

FROM python:3.12.11-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 libgl1 libglib2.0-0 libfontconfig1 libfreetype6 liblz4-1 libzstd1 ca-certificates && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 bot && useradd -u 10001 -g bot -m bot \
    && mkdir -p /var/lib/shuiyuan /var/tmp/shuiyuan && chown -R bot:bot /var/lib/shuiyuan /var/tmp/shuiyuan
COPY --from=postgres:17.6-bookworm /usr/lib/postgresql/17/bin/pg_dump /usr/local/bin/pg_dump
COPY --from=postgres:17.6-bookworm /usr/lib/postgresql/17/bin/pg_restore /usr/local/bin/pg_restore
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp/shuiyuan
WORKDIR /app
USER 10001:10001
CMD ["shuiyuan-bot", "--config", "/etc/shuiyuan/deployment.toml", "--profile", "remote", "--web"]
