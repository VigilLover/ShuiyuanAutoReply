FROM postgres:17.6-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential postgresql-server-dev-17 && rm -rf /var/lib/apt/lists/*
COPY deploy/vendor/pgvector /build/pgvector
RUN cd /build/pgvector && make OPTFLAGS="" && make install
FROM postgres:17.6-bookworm
COPY --from=builder /usr/lib/postgresql/17/lib/vector.so /usr/lib/postgresql/17/lib/
COPY --from=builder /usr/share/postgresql/17/extension/vector* /usr/share/postgresql/17/extension/
COPY deploy/postgres/init.sh /docker-entrypoint-initdb.d/10-shuiyuan.sh
