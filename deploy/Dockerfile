# syntax=docker/dockerfile:1
FROM node:22.18.0-bookworm-slim@sha256:752ea8a2f758c34002a0461bd9f1cee4f9a3c36d48494586f60ffce1fc708e0e AS frontend
WORKDIR /build/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:0.8.15@sha256:a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0 AS uv
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra server --no-install-project --no-editable
COPY src ./src
COPY README.md ./
COPY --from=frontend /build/src/shuiyuan_auto_reply/interfaces/api/static ./src/shuiyuan_auto_reply/interfaces/api/static
RUN uv sync --frozen --no-dev --extra server --no-editable
RUN .venv/bin/python -c "import importlib.util; assert all(importlib.util.find_spec(m) is None for m in ('torch','sentence_transformers','neo4j','neomodel'))"

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 libgl1 libglib2.0-0 libfontconfig1 libfreetype6 liblz4-1 libzstd1 ca-certificates && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 bot && useradd -u 10001 -g bot -m bot \
    && mkdir -p /var/lib/shuiyuan /var/tmp/shuiyuan && chown -R bot:bot /var/lib/shuiyuan /var/tmp/shuiyuan
COPY --from=postgres:17.6-bookworm@sha256:f3bd19c606e442c3d7bdfa8002e03fe260a1023351e0ea4598032022b68dd6e3 /usr/lib/postgresql/17/bin/pg_dump /usr/local/bin/pg_dump
COPY --from=postgres:17.6-bookworm@sha256:f3bd19c606e442c3d7bdfa8002e03fe260a1023351e0ea4598032022b68dd6e3 /usr/lib/postgresql/17/bin/pg_restore /usr/local/bin/pg_restore
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp/shuiyuan
WORKDIR /app
USER 10001:10001
CMD ["shuiyuan-bot", "--config", "/etc/shuiyuan/deployment.toml", "--profile", "remote", "--web"]
