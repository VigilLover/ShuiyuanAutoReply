FROM ghcr.io/astral-sh/uv:0.8.15@sha256:a5727064a0de127bdb7c9d3c1383f3a9ac307d9f2d8a391edc7896c54289ced0 AS uv
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build
COPY deploy/mcp/requirements.lock ./
RUN uv venv /opt/venv && uv pip sync --python /opt/venv/bin/python requirements.lock --require-hashes
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
RUN groupadd -g 10002 mcp && useradd -u 10002 -g mcp -m mcp && mkdir -p /var/tmp/mcp && chown mcp:mcp /var/tmp/mcp
COPY --from=builder /opt/venv /opt/venv
COPY deploy/vendor/simplemcp /srv/mcp
COPY deploy/mcp/fetch.py deploy/mcp/server.py /srv/deploy/
ENV PATH="/opt/venv/bin:$PATH" PYTHONPATH=/srv/mcp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/var/tmp/mcp
WORKDIR /srv/deploy
USER 10002:10002
CMD ["python", "server.py"]
