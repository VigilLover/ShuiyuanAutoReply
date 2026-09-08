#!/bin/sh
set -eu
# psql variables quote role passwords as SQL literals, never interpolate shell into SQL.
psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --set=app_password="$(cat /run/secrets/database_app_password)" <<'SQL'
CREATE ROLE shuiyuan_app LOGIN PASSWORD :'app_password';
REVOKE ALL ON DATABASE shuiyuan FROM PUBLIC;
GRANT CONNECT ON DATABASE shuiyuan TO shuiyuan_app;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO shuiyuan_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT,INSERT,UPDATE,DELETE ON TABLES TO shuiyuan_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE,SELECT ON SEQUENCES TO shuiyuan_app;
SQL
