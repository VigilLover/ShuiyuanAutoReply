# 本地兼容与低资源远程部署

本方案使用同一分支和同一 TOML 格式。本地保留 m3e-base/Neo4j；远程使用
`qwen3.7-text-embedding` 1024 维、PostgreSQL/pgvector，以及独立的 SimpleMCP。
远程 Bot 和 Web 同进程，默认最多执行 3 个任务。同一会话串行，图片及网页抓取各串行。
目标服务器是 Ubuntu Server 24.04 LTS、x86_64、2 vCPU、2GB RAM、40GB SSD。
这是需要实测验收的容量目标，不是当前代码的性能保证。

## 1. 目录和配置规则

- `config/deployment.example.toml`：公共项、本地项和远程项的完整起步配置。
- `deploy/compose.yaml`：生产部署；`deploy/compose.test.yaml`：无生产凭据的隔离验证。
- `scripts/deploy/`：源码准备、凭据初始化、顺序启动、集成与持续运行检查。
- `shuiyuan-ops`：配置检查、数据库迁移、数据导入导出、Cookie 转换和备份恢复。

从仓库根目录执行本文命令。操作命令的 `--config`、`--profile` 必须放在子命令前。

```bash
cp config/deployment.example.toml config/deployment.toml
```

优先级为：显式 CLI 参数 > 选中 profile > common > 旧环境变量 > 默认值。
未传 `--config` 时继续加载原有 `.env`；默认 profile 为 `local`。
配置中的文件路径相对 TOML 文件所在目录解析，绝对路径保持原样。

密钥字段支持 `{env="ENV_NAME"}` 或 `{file="path/to/secret"}`；引用缺失则启动失败。
这些字段不能同时使用两种引用。示例远程配置的 `/run/secrets/...` 是容器内路径，
因此远程配置检查应在容器内运行，而不是直接在本机运行。

`providers` 中的键是现有 Provider 环境变量名，例如 `DEEPSEEK_API_KEY`、
`IMAGE_GEN_API_KEY`、`IMAGE_GEN_API_URL`、`OPENROUTER_API_KEY`。
部署脚本初始化 DeepSeek、Embedding 和图片生成 Key；Compose 已为 Bot/migrate 挂载 `image_key`，
配置示例使用 `IMAGE_GEN_API_KEY={file="/run/secrets/image_key"}`。
不使用生图时可留空，脚本仍创建空 secret 文件以满足 Compose 挂载；生图功能需有效 Key 才能调用。
不要在构建参数、Dockerfile 或 Git 中放入真实 Key。

基础设施配置在重启后生效。Web 中的 Prompt、聊天 Key 和工具配置继续支持热切换，
但不会修改 Embedding、数据库或 Compose 资源限制。Web 保存的聊天 Key 优先于该渠道的环境 Key；
如果轮换文件后旧 Key 仍生效，需要在 Web 设置中更新对应已保存 Key。

## 2. 保留本地运行

```bash
uv sync --frozen --extra dev --extra server --extra local-embedding --extra neo4j
uv run --no-sync shuiyuan-bot
uv run --no-sync shuiyuan-bot wolf_lumine --web
uv run --no-sync shuiyuan-api
uv run --no-sync shuiyuan-ops --config config/deployment.toml --profile local config check
```

本地 `.env` 中保留原数据库、模型、MCP 等设置。配置示例明确选择 m3e-base、768 维和 Neo4j。
本地模型只在需要时加载；远程镜像不安装该 extra。
如果单独使用 `pip`，安装 `pip install -e '.[server,local-embedding,neo4j]'`。
新增依赖拆分后，单纯 `pip install -e .` 不再包含本地模型与 Neo4j SDK。

本地旧 Cookie pickle 仍可读取，但只能使用自己生成且可信的文件。远程禁止 pickle。
本地默认允许兼容的数据库自动初始化；远程只能通过迁移命令初始化 schema。

## 3. 服务器与 Docker

新建服务器时选择 Ubuntu Server 24.04 LTS 的 amd64 镜像，分配 2 vCPU、2GB 内存和
40GB SSD 系统盘，不安装桌面环境。使用云控制台注入 SSH 公钥，创建具有 sudo 权限的部署账号，
登录后更新系统包并按系统提示重启。保留云控制台访问能力，确认 SSH 公钥登录正常后再调整 SSH 设置。

使用 Docker 官方 Ubuntu 仓库安装 Docker Engine、Buildx 和 Compose 插件，步骤参考：
[Docker Ubuntu 安装文档](https://docs.docker.com/engine/install/ubuntu/)。
依次检查冲突的旧 Docker/containerd 包、安装官方签名密钥、添加对应 Ubuntu 版本的 apt 源，
再安装 `docker-ce`、`docker-ce-cli`、`containerd.io`、`docker-buildx-plugin` 和
`docker-compose-plugin`。使用文档中的仓库安装步骤，安装后运行 `hello-world` 验证。
安装后检查：

```bash
docker version
docker compose version
docker info
```

Docker 用户组具有很高的宿主机权限，仅授予部署管理员。宿主机不需要安装 Python 项目依赖或 Node；
执行辅助脚本需要 Python 3。SSH 使用密钥登录，云安全组限制 SSH 来源。
Bot、模型 API 与 MCP 搜索需要出站网络；仅数据库网络完全内部隔离。

默认只映射 `127.0.0.1:11451`。不映射 PostgreSQL 5432 或 MCP 58000。
Docker 发布端口可能绕过 UFW 的预期过滤，因此不要把端口映射改为裸 `11451:11451`。
参见 <https://docs.docker.com/engine/network/packet-filtering-firewalls/>。

可配置 1–2GB swap 作为偶发峰值缓冲，但持续换页不能作为正常运行状态。
保持至少约 8GB 可用磁盘用于日志、数据库临时文件和镜像升级；配置备份到其他机器或对象存储。

## 4. Cookie、模型 Key 和数据库密码

### 4.1 获取并转换 Cookie

在可信本机使用仓库已有 `get_cookies.ipynb` 完成正常 jAccount 登录。不得把交互登录、账号密码
或 Notebook 输出放进镜像。确认 Cookie 对应配置的 Bot 用户名。

```bash
uv run --no-sync shuiyuan-ops cookie convert cookies /tmp/forum-cookie.json --trust-pickle
```

`--trust-pickle` 表示明确允许读取你自己的可信 pickle；恶意 pickle 可以执行代码。
输出文件以 0600 创建，存在时拒绝覆盖。远程 JSON 格式为：

```json
{"version":1,"domain":"shuiyuan.sjtu.edu.cn","cookies":{"COOKIE_NAME":"COOKIE_VALUE"}}
```

不要将真实值贴入工单或提交。该格式将 Cookie 限定用于水源社区，不是通用浏览器 Cookie 导入格式。

### 4.2 初始化部署凭据

在准备部署目录的可信机器上执行：

```bash
python3 scripts/deploy/init_secrets.py --cookie /tmp/forum-cookie.json
```

脚本隐藏输入 Embedding、DeepSeek 和图片生成 Key，自动生成不同的 PostgreSQL 管理员及运行账号密码；
不会打印真实凭据，也不会覆盖已有文件。

已有部署仅补充缺失的生图 Key 时执行 `python3 scripts/deploy/init_secrets.py --image-key-only`，
不会改动已有数据库密码或其他 Key。如果 `secrets/image_key` 已存在则拒绝覆盖。
更新 Compose/config 后执行 `docker compose -f deploy/compose.yaml up -d --force-recreate bot`，
无需重新构建镜像。

`secrets/` 是 0700 私有目录；里面的文件为 0444，以便 Docker 将单个 secret 挂载给不同容器 UID。
不要将该目录本身改成公开可读。普通 Compose 文件型 secret 不代表宿主机加密存储；
有权控制 Docker 或宿主机 root 的人仍能读取密钥。

凭据目录与真实配置均被 Git 和 Docker 构建上下文忽略。使用 SSH/SCP 安全传输到服务器，
传输后检查目录权限，不要使用命令参数直接传递 Key。

### 4.3 百炼 Embedding 设置

在 `[profiles.remote.embedding]` 填入实际 OpenAI 兼容 `base_url`，删除
`YOUR_WORKSPACE_ID` 占位符。Key、地域和业务空间必须匹配控制台；不要把控制台网页 URL 当成 API 地址。

默认模型为 `qwen3.7-text-embedding`、1024 维，接口调用为 `/embeddings`，原始字符串输入，
输出要求 `float`。客户端每批默认 10 条、并发 2、单次请求超时 30 秒，最多 3 次尝试。
对 429、5xx 和连接故障退避，认证或参数错误不重试；超长输入由服务端明确拒绝，不静默截断。
模型的批量上限通过 `max_batch_size` 配置。不要使用另一模型的 tokenizer 计算后截断文本。

模型文档：<https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api>。
当前文档列出该模型最多 20 条输入、单条最多 128,000 Token。部署前确认账号可用性和配额。

## 5. 构建、发布和传输镜像

推荐在开发机或 CI 构建，不在 2GB 生产服务器上运行 Node 构建或安装本地 ML 依赖。
先准备锁定的构建源码：

```bash
python3 scripts/deploy/prepare_sources.py
```

也可以从现有本机 MCP 仓库获取同一个固定提交：

```bash
python3 scripts/deploy/prepare_sources.py --mcp-source /absolute/path/to/SimpleMCP-for-ShuiyuanAutoReply
```

脚本只写 `deploy/vendor/` 下的独立构建副本，不修改原仓库。固定版本是：

- SimpleMCP：`de42d48eb81644604ebade8524af2748c4cc3e6b`
- pgvector 0.8.2：`cab9da72c04353f143bb06b42ab70a403daac64a`

版本无法取得、已有副本被修改时会失败，不会自动换成最新版本。
数据库镜像从 PostgreSQL 17.6 构建 pgvector；应用和 MCP 使用固定 Python、Node、uv 版本。
Python 应用使用 `uv.lock`，MCP 使用含哈希的独立依赖清单。

Linux x86_64 目标、包括在 Apple Silicon 上构建时：

```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose -f deploy/compose.yaml build postgres bot mcp
```

本地镜像默认标签见 Compose。可通过 `BOT_IMAGE`、`POSTGRES_IMAGE`、`MCP_IMAGE` 指定你控制的仓库标签。
发布时使用唯一版本标签，并记录 registry 返回的 digest；生产可设置 `BOT_IMAGE=registry/image@sha256:...`。
固定版本标签仍可能被镜像仓库重新指向，digest 才是最终发布的不可变标识。
不要把示例版本当作永久免维护版本，应周期性扫描漏洞并在验证后更新。

没有镜像仓库时：

```bash
docker save -o /tmp/shuiyuan-images.tar shuiyuan-bot:remote-v1 shuiyuan-postgres:17.6-vector0.8.2 shuiyuan-mcp:de42d48
scp /tmp/shuiyuan-images.tar YOUR_SERVER:/tmp/
# 服务器上：
docker load -i /tmp/shuiyuan-images.tar
```

同时传输 `deploy/`（可不传 vendor）、`scripts/deploy/`、`config/deployment.toml` 和 `secrets/`，
保持仓库相对目录结构。应用运行时不读取宿主机源代码目录。

## 6. 首次初始化及数据迁移

### 6.1 冻结本地写入并导出

在本地停止旧 Bot 及任何会写长期记忆的网页 Runtime。不要让两个 Bot 同时监听同一账号。

```bash
mkdir -p transfer
uv run --no-sync shuiyuan-ops --config config/deployment.toml --profile local corpus export transfer/corpus.jsonl --persona wolf_lumine
uv run --no-sync shuiyuan-ops --config config/deployment.toml --profile local memory export transfer/memory.jsonl
```

人物语料默认从 Neo4j 只读导出；没有 `userid` 的旧节点会拒绝导出，需要先补齐人物归属。
也可从原始 CSV 导出：

```bash
uv run --no-sync shuiyuan-ops --config config/deployment.toml --profile local corpus export transfer/corpus.jsonl --csv user_archive/wolf_lumine/user_archive.csv --persona wolf_lumine
```

CSV 支持 `post_raw` 或 `text` 列，过滤自动回复、签名和空内容。输出文件存在时拒绝覆盖。
长期记忆按 namespace/key/value 导出，附带源时间戳；源数据库不会执行 schema 修改。
迁移前保存源库备份。

### 6.2 启动数据库并执行迁移

以下在服务器执行。先检查清单，再仅启动数据库：

```bash
docker compose -f deploy/compose.yaml config --quiet
docker compose -f deploy/compose.yaml up -d --wait postgres
docker compose -f deploy/compose.yaml run --rm migrate
```

迁移容器挂载管理员 DSN，运行 Bot 只挂载 `shuiyuan_app` DSN。
迁移初始化 pgvector、业务表、LangGraph Store、向量空间指纹，以及 SQLite/论坛队列。
运行账号只获得连接、schema 使用及所需表/序列 DML 权限，不使用 PostgreSQL 超级用户。
`docker-entrypoint-initdb.d` 的角色创建只在新 PostgreSQL 数据卷首次启动时运行。
已有数据库不能靠更改 secret 文件完成数据库密码轮换。

远程目标必须是新数据库，不要把本地 768 维旧表直接拷贝进去。未登记向量空间的既有记忆库会被拒绝。
服务标识、模型、维度、预处理版本构成向量空间指纹；同样是 1024 维也不代表不同模型可以混用。

### 6.3 导入并重新生成向量

把 `transfer/` 安全传到服务器。目录需允许容器 UID 10001 写入断点和错误报告：

```bash
sudo chown -R 10001:10001 transfer
sudo chmod 700 transfer
```

先仅验证输入，不发送付费 Embedding 请求：

```bash
docker compose -f deploy/compose.yaml run --rm -v "$PWD/transfer:/transfer" bot shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote corpus import /transfer/corpus.jsonl --dry-run
docker compose -f deploy/compose.yaml run --rm -v "$PWD/transfer:/transfer" bot shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote memory import /transfer/memory.jsonl --dry-run
```

确认输入后正式导入，以下两条会调用外部 Embedding 并产生费用：

```bash
docker compose -f deploy/compose.yaml run --rm -v "$PWD/transfer:/transfer" bot shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote corpus import /transfer/corpus.jsonl
docker compose -f deploy/compose.yaml run --rm -v "$PWD/transfer:/transfer" bot shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote memory import /transfer/memory.jsonl
```

迁移保留人物归属、记忆 namespace/key/value；新库内创建/更新时间为重新导入时间，源时间戳留在导出包。
原始文本会发送给配置的外部 Embedding 服务。

导入产生 `.checkpoint.json` 和 `.failures.json`，报告只记录行号及错误类型。
每处理 10 条输出一次进度；完整成功后输出本次处理数量。
中断后运行相同命令继续；重复人物语料按内容去重，记忆按 namespace/key 更新。
断点绑定源文件校验和、向量空间和目标库 generation；不能把旧库断点当作新建库的完成记录。
如果源文件修改，应保存旧断点、使用新的输入文件名重新导入；未变的人物语料会跳过重复计算。
记忆导入需要足够数据库权限，但不能因此改成生产 Bot 使用管理员账号。

本次不迁移 SQLite 会话、旧图片、运行轨迹和 UI 保存的聊天 Key。
旧本地数据库及 Neo4j 保留，不能通过删除旧数据来节省新服务器空间。

## 7. 预检、启动与 SSH 管理

```bash
docker compose -f deploy/compose.yaml run --rm bot shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote config check
docker compose -f deploy/compose.yaml run --rm bot shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote doctor --probe-forum
```

`config check` 只输出脱敏配置；`doctor` 默认检查 Cookie 文件与数据库，不发帖。
`--probe-forum` 只读验证社区实际登录用户名；如需验证模型连通性，显式追加 `--probe-embedding`，会产生一次模型调用。
预检不能用“社区首页返回 200”替代真实身份检查。

正式启动：

```bash
python3 scripts/deploy/start.py
```

脚本等待 PostgreSQL 健康，停止既有 Bot，完成迁移后再启动 Bot/MCP。
生产环境采用该脚本；单独执行 `docker compose up` 不会自动替你完成迁移。
如需从 registry 拉取镜像，使用 `--pull`。
禁用 MCP 时同时将配置 `mcp.enabled=false` 并使用 `--no-mcp`；已有 MCP 可执行 `docker compose ... stop mcp` 停止。

初次启动将当前通知设为基线，不批量回复历史；之后重启会从持久化游标补拉。
未完成任务恢复，发帖中断任务标记为 `needs_review`，不自动重发。

在管理员本机建立 SSH 隧道：

```bash
ssh -N -L 11451:127.0.0.1:11451 YOUR_SERVER
```

浏览器访问 `http://127.0.0.1:11451`。本次没有公网登录系统，不要把该端口直接公开。
管理页面启动不依赖社区登录成功；Bot 登录失败时保留 Web 并退避重试。

```bash
curl http://127.0.0.1:11451/api/live
curl http://127.0.0.1:11451/api/runtime-health
docker compose -f deploy/compose.yaml logs --tail=100 bot
docker compose -f deploy/compose.yaml ps
docker stats --no-stream
```

`/api/live` 只表示进程可以响应；`/api/runtime-health` 返回数据库、最近轮询与任务状态。
旧 `/api/health` 为兼容接口，不能作为完整 Bot 就绪检查。
Compose 的 unhealthy 本身不会自动重启容器；异常退出由 restart 策略恢复，外部故障应结合健康信息告警。

## 8. 资源限制和运行边界

- 应用 800MiB、PostgreSQL 384MiB、MCP 256MiB，保留宿主机余量；实际峰值仍需测试。
- 应用最大并发 3，同会话 FIFO；排队默认上限 100，满载网页请求返回繁忙。
- 默认轮询间隔 5 秒，整轮超时 15 分钟，停机排空最多 60 秒，Compose 停止宽限 75 秒。
- 同时只解码/转换一张图片，单张 20MiB、整轮 40MiB、最多 20 张、最多 400 万像素，输出长边 2048。
- 超大动画缩放时保留首帧；超像素图片明确拒绝，不尝试先完整解码后压缩。
- 闲置会话缓存最多 100 个、TTL 30 分钟，论坛/网页从持久化会话读取历史。
- 图片保留 30 天、配额 3GiB，写入前清理过期图片，超配额拒绝新写入；过期图片链接可能返回 404。
- 应用/MCP 临时目录使用磁盘卷而非大型 tmpfs。日志单文件 10MB、保留 3 份。

监控同时关注宿主机 MemAvailable、容器内存、swap、磁盘空间、队列积压与回复延迟。
磁盘告警阈值建议 8GB；当前项目提供健康/状态入口，通知渠道由运维接入，不会自动发送外部告警。

SimpleMCP 不连接数据库网络、不挂载 Bot Key/Cookie、Docker socket 或宿主机 `/proc`。
网页抓取仅允许实际连接到公网 IP 的 HTTP(S) 80/443，最多 5 次重定向、30 秒、5MiB 流式上限。
为避免压缩内容绕过资源上限，抓取包装要求 identity 编码，拒绝服务器强制压缩的响应。
该包装保持工具名称和参数，但对不符合安全限制的网站会返回失败；不会退回原 uvx 抓取绕过限制。
硬件工具仅看到容器可见的系统信息。时间、搜索、图片搜索沿用固定上游工具实现。

## 9. 故障处理、凭据轮换和人工核对

- `Vector space mismatch`：配置与库不一致；恢复匹配配置或使用新数据库重新向量化，不修改旧表维度硬凑。
- `referenced environment variable is missing`/文件缺失：检查 secret 引用和挂载路径，避免把值打印到日志。
- Cookie/用户名错误：重新在可信设备登录并转换，替换 secret，重新创建 Bot 容器；对单文件 bind mount，原子替换宿主机文件后必须重建容器使挂载指向新文件。
- 模型 401/403：检查业务空间、地域与 Key；429：检查配额与并发；参数错误不自动无限重试。
- PostgreSQL 不可用：检查健康、磁盘、凭据和指纹。禁止关闭 fsync、WAL 或 autovacuum 换取资源。
- MCP 缺失：检查 service 是否运行、内部地址 `http://mcp:58000/sse`，以及是否同时禁用了配置和服务。
- OOM：先看图片和 MCP 峰值、是否启动了额外 Runtime/开发服务器，再降低配置并发到 2；不得以无限 swap 隐藏持续内存超限。

数据库密码轮换需要先在 PostgreSQL 中更新对应角色，再同步 DSN/密码 secret 并重建容器；
单纯编辑初始化密码文件不会改变已存在的数据卷。使用交互式 `psql \password` 或安全管理渠道，避免命令行明文。

`needs_review` 表示可能已发帖但本地未确认。先在社区查询该话题的实际回复，并核对 Bot 用户及时间，
再更新队列状态。不要直接把所有任务重新置为 pending。运维可在停机备份后用 SQLite 工具只查看
`forum_jobs` 的 username/post_id/status/reply_id，确认已发帖的标记 sent，确定未发帖的才手动重置 pending。
`failed` 任务同样应先核对错误及外部副作用，修复后按单条事件恢复。

## 10. 备份、恢复和升级

停止所有写入者后备份，包括 Bot/Web 及任何导入任务：

```bash
docker compose -f deploy/compose.yaml stop bot
mkdir -p backups
sudo chown 10001:10001 backups
sudo chmod 700 backups
docker compose -f deploy/compose.yaml run --rm -v "$PWD/backups:/backups" bot shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote backup /backups/2026-09-07 --writers-stopped
```

备份包含 PostgreSQL 自定义格式 dump、SQLite 一致性备份、`master.key`、图片及其他状态文件。
SQLITE 主文件不能单独热复制。`master.key` 丢失后无法解密原 UI 密钥。
manifest 只在备份完成后生成。备份包含敏感数据，应加密存储、传到异机并定期恢复演练。
配置及原始 secret 需另行安全保存，不能仅依赖数据备份恢复所有登录信息。

恢复必须使用**空目标数据库和空状态卷**，停用所有写入者，使用匹配的向量空间配置与应用版本。
`restore` 使用 PostgreSQL 客户端返回错误时立即失败，不会自动删除现存数据：

```bash
# 在准备好的新目标实例/空卷上，用 migrate 服务提供的管理员 DSN 恢复：
docker compose -f deploy/compose.yaml run --rm -v "$PWD/backups:/backups:ro" migrate shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote restore /backups/2026-09-07 --writers-stopped
```

恢复后检查 schema、权限、指纹及健康，再启动单个 Bot。恢复到其他容器路径前应保持
`/var/lib/shuiyuan` 不变，因为图片记录包含绝对路径。

升级顺序：备份 → 拉取已验证的新镜像 → 停 Bot → 迁移 → 启动 → 检查健康和队列。
保留上一版本的镜像 digest 和匹配备份。不要使用 `docker compose down -v` 升级，它会删除数据卷。
不要在无人值守情况下自动升级 PostgreSQL 主版本。

回滚须同时考虑 schema 和向量空间，不能只把旧代码连接到新模型向量库。
回切本地前先停止远程 Bot；远程期间新增的长期记忆若要回本地，需要再导出并用 m3e-base 重新向量化，不能复制远程向量。

## 11. 测试和验收

```bash
uv run --no-sync pytest -q
npm --prefix web run build
docker compose -f deploy/compose.yaml config --quiet
docker compose -f deploy/compose.test.yaml config --quiet
```

默认测试不联系真实社区或付费模型；需要真实服务的测试必须显式 `--run-live`。
镜像构建阶段断言没有安装 torch、sentence_transformers、neo4j、neomodel。

构建完成后运行隔离集成栈，使用临时数据库和合成模型，不读取生产 secret，不发帖：

```bash
python3 scripts/ci/integration.py
# 清理的是 shuiyuan-validation 项目，不是生产项目：
docker compose -f deploy/compose.test.yaml down -v
```

该脚本验证迁移、重复导入、人物隔离、长期记忆导出、图片保存、三并发执行、队列恢复和 MCP 时间工具。
模型采用确定性假向量，不代表真实模型质量或社区网络行为。

在目标 2GB Linux 环境进行 24 小时合成持续运行：

```bash
VALIDATION_SECONDS=86400 python3 scripts/ci/integration.py
```

同时从宿主机采集 `docker stats`、MemAvailable、swap、磁盘与容器重启/OOM 记录。
要求无 OOM、无持续内存增长、至少约 200MiB 可用余量；实际模型/社区的受控验收另行显式执行。
生产网络验收应覆盖一条受控提及、长任务与短任务并发、Cookie 失效、数据库重启、停机补拉和备份恢复。

2026-09-08 已在 Docker Desktop 的 Linux/amd64 容器中完成三镜像构建、真实 PostgreSQL
运行账号权限、迁移与备份恢复、模拟论坛的 Bot/Web 启动验收；没有完成目标 2GB 服务器容量
测试或 24 小时持续运行。本机容器测试不能替代目标服务器验收。CI/CD 见 [cicd.md](cicd.md)。

本次开发验证记录（2026-09-07）：离线 pytest 为 154 passed、20 skipped；本次涉及的
52 个 Python 文件通过 Black/isort 检查，前端构建、两套 Compose 静态配置检查通过。
跳过项为显式 opt-in 的外部服务测试，没有使用真实模型计费或向社区发帖。
备份恢复演练、真实 Cookie 轮换、目标服务器重启和停机补拉验收仍待部署环境执行。
