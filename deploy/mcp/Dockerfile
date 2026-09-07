FROM ghcr.io/astral-sh/uv:0.8.15 AS uv
FROM python:3.12.11-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build
COPY deploy/mcp/requirements.lock ./
RUN uv venv /opt/venv && uv pip sync --python /opt/venv/bin/python requirements.lock --require-hashes
FROM python:3.12.11-slim-bookworm
RUN groupadd -g 10002 mcp && useradd -u 10002 -g mcp -m mcp && mkdir -p /var/tmp/mcp && chown mcp:mcp /var/tmp/mcp
COPY --from=builder /opt/venv /opt/venv
COPY deploy/vendor/simplemcp /srv/mcp
COPY deploy/mcp/fetch.py deploy/mcp/server.py /srv/deploy/
ENV PATH="/opt/venv/bin:$PATH" PYTHONPATH=/srv/mcp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/var/tmp/mcp
WORKDIR /srv/deploy
USER 10002:10002
CMD ["python", "server.py"]
