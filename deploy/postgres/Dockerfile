FROM postgres:17.6-bookworm@sha256:f3bd19c606e442c3d7bdfa8002e03fe260a1023351e0ea4598032022b68dd6e3 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential postgresql-server-dev-17 && rm -rf /var/lib/apt/lists/*
COPY deploy/vendor/pgvector /build/pgvector
RUN cd /build/pgvector && make OPTFLAGS="" && make install
FROM postgres:17.6-bookworm@sha256:f3bd19c606e442c3d7bdfa8002e03fe260a1023351e0ea4598032022b68dd6e3
COPY --from=builder /usr/lib/postgresql/17/lib/vector.so /usr/lib/postgresql/17/lib/
COPY --from=builder /usr/share/postgresql/17/extension/vector* /usr/share/postgresql/17/extension/
COPY deploy/postgres/init.sh /docker-entrypoint-initdb.d/10-shuiyuan.sh
