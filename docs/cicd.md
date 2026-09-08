# CI/CD 设置、发布和恢复

首次操作从 [逐步部署指南](first-deployment.md) 开始；Fork 的主线切换与上游同步见 [分支策略](branch-strategy.md)。

本项目采用 GitHub Actions → GHCR → 手动触发 SSH 部署。发布代码来自 `main`；部署工作流只能从默认分支 `main` 手动运行。日常 CI 不访问真实论坛或付费模型，测试容器只有内部网络。

## 1. 工作流与首次启用

- CI：PR 到 dev/main，以及 dev/main/remote-deploy 推送；Python 3.12 的 remote/local 两套依赖组合、全量 Black/isort、离线 pytest、Node 22 前端构建和 Compose 静态检查。
- Release：`vX.Y.Z` 标签；验证 main 历史归属，重新执行 CI，构建 linux/amd64 三个镜像，集成通过后推送同一镜像并发布部署包。
- Deploy production：手动输入正式版本与 deploy/rollback，使用 production Environment，经固定 SSH 入口更新服务器。

产品代码、工作流和依赖脚本统一通过 PR 合入 main，main 同时作为默认分支、正式发布来源和手动部署入口。dev 保留为开发集成分支，可继续通过 PR 向 main 提交变更；不再单独向 main 复制工作流。首次切换时先合并 dev → main，等待 CI 成功后再打正式标签。本文说明不会自动执行合并、推送或仓库设置。

在 GitHub Settings → Rules → Rulesets 为 main 设置 PR 合并规则（dev 可同时保留检查），首次运行后选择 CI 的 `Python (remote)`、`Python (local)`、`frontend` 为必需检查。为正式版本标签设置创建/更新权限限制，禁止移动已发布标签。

在 Settings → Actions 设置允许工作流使用官方 Actions；固定 SHA 已在实施时核验，基础镜像使用 registry digest。定期通过独立 PR 更新，而不是运行时自动升级。启用 GHCR package 发布权限，Release job 仅获得所需 contents/packages 写权限；PR 只有 contents 读权限。

uv 使用 `--locked`，锁文件不匹配即失败。不要在 CI 自动更新锁文件。依赖调整应连同经过审阅的锁文件变更一起提交；CI 只使用仓库提交内容。

GitHub 参考：[镜像发布](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)、[构建缓存](https://docs.docker.com/build/ci/github-actions/cache/)。uv 下载缓存目前依赖安装环境，Docker 构建和 npm 配置了缓存；首次运行耗时取决于下载，不承诺固定分钟数。

## 2. 发布版本

先确认目标提交已经合入 main 并通过 CI，然后由维护者执行：

```bash
git fetch origin
git tag v1.0.0 origin/main
git push origin v1.0.0
```

这里只是操作示例，不自动执行。标签必须是无前导零的三段数字正式版本，不接受 latest、分支或预发布字符串。

Release 在构建机生成三个镜像：`ghcr.io/vigillover/shuiyuan-bot`、`shuiyuan-postgres`、`shuiyuan-mcp`，附版本和完整 Git SHA 标签。部署使用 digest。固定 namespace 为本仓库所有者，fork 不应直接发布到该 namespace。

`deploy/release-policy.json` 必须随发布审阅：

- `none`：普通发布不执行迁移；schema_id 不同则部署端拒绝。
- `backward-compatible`：填写 `compatible_from` 正式版本列表；集成测试拉取每个旧应用镜像，在新 schema 上读取测试语料、长期记忆和 SQLite 状态，成功后将旧版本镜像 digest 写入兼容证据。
- `manual`：可以构建发布，但普通部署入口拒绝，需要单独维护。

schema_id 保守覆盖持久化、迁移、检索、数据库代码和依赖声明/锁文件；格式或依赖修改也可能要求兼容测试。该机制不能代替迁移审阅，新增持久化实现时必须扩展 identity 覆盖范围。当前初始化仍使用项目已有迁移入口，没有自动生成数据库降级脚本。

三个镜像分别使用缓存。数据库/MCP 版本变化不是普通应用发布的一部分；相同构建输入应复用既有缓存并核对最终 digest。若重建产生了新的 PostgreSQL digest，部署端会拒绝，必须保持发布清单引用既有数据库 digest，或走维护流程。

部署包包含 release.json、Compose、脚本、配置示例、文档、测试摘要及 SHA256SUMS；不含真实配置、Cookie、Key、数据或模型权重。GitHub Release 在所有上传成功后才从 draft 转为正式。失败留下的 draft 由维护者检查后清理，不覆盖已发布版本。

## 3. 服务器初始化

服务器使用 Ubuntu 24.04 amd64、Python 3.12 和 Docker Compose，目录固定为 `/opt/shuiyuan`。按 deployment.md 完成 Docker 安装、首次数据库初始化与重新向量化。

先在可信设备下载、核验正式部署包；以本地管理员身份安装受信控制器：

```bash
sudo python3 scripts/deploy/install_controller.py
```

控制器安装到 `/usr/local/lib/shuiyuan`；上传的新版本不会替换受信控制器。控制器升级需管理员在服务器显式重跑安装程序。

将真实 TOML 放在 `/opt/shuiyuan/shared/deployment.toml`，秘密文件放在 `shared/secrets`，保持 secret 文件名与 Compose 对齐。shared/secrets 目录 0700；单文件 0444 供非 root 容器通过单文件 bind mount 读取。Compose secrets 不提供宿主机静态加密。

私有 GHCR 镜像需要服务器管理员进行 `docker login ghcr.io`，使用只有 read:packages 权限的凭据；凭据保存在执行控制器的 root Docker 配置中。接收部署包时，服务器还会独立向 GitHub API 核对正式 Release 的 asset digest，不只相信 SSH 发送者提供的校验和。私有代码仓库另需将只有该仓库 Contents 只读权限的 GitHub token 保存到 `/opt/shuiyuan/shared/github_read_token`（root:root、0600）；公开仓库可以省略，但仍受未认证 API 限流影响。GitHub API 不可用、包没有 digest 或校验失败时停止接收，不退回信任上传者。不要将仓库写令牌放到服务器。

创建专用 SSH 用户 `shuiyuan-deploy`，不加入 docker 组，不授予普通 sudo。由管理员把 GitHub 部署公钥写入 root 管理的 authorized_keys，并使用以下限制：

```text
restrict,command="sudo -n /usr/local/sbin/shuiyuan-ssh" ssh-ed25519 PUBLIC_KEY
```

通过 `visudo -f /etc/sudoers.d/shuiyuan-deploy` 配置：

```text
Defaults:shuiyuan-deploy env_keep += "SSH_ORIGINAL_COMMAND"
shuiyuan-deploy ALL=(root) NOPASSWD: /usr/local/sbin/shuiyuan-ssh ""
```

验证 sudoers 语法，并保持 authorized_keys 与受信控制器不可由部署账号修改。入口仅接受 receive/version/checksum、deploy/version、rollback/version、status，拒绝任意 shell、SCP、SFTP 和交互命令。容器部署能力本身属于高权限，只有可信维护者可修改正式发布工作流和 Compose。

GitHub production Environment 添加：`DEPLOY_HOST`、`DEPLOY_USER`、`DEPLOY_SSH_KEY`、`DEPLOY_KNOWN_HOSTS`。只支持 SSH 22 端口和主机名/IPv4 地址。KNOWN_HOSTS 必须通过可信服务器控制台核验，不能在部署任务中临时 ssh-keyscan 后盲目信任。若仓库套餐支持，可增加 Environment reviewer；手动触发是基础控制，不依赖该功能可用。

### 首个正式版本登记

控制器不会自动初始化数据库。以下以首次发布 `v1.0.0` 为例，在服务器本地管理员终端执行。先从 GitHub 正式 Release 下载部署包和 SHA256SUMS，再核验并接收：

```bash
sha256sum -c SHA256SUMS
bundle_sha=$(sha256sum shuiyuan-v1.0.0.tar.gz | cut -d ' ' -f 1)
sudo env SSH_ORIGINAL_COMMAND="receive v1.0.0 $bundle_sha" /usr/local/sbin/shuiyuan-ssh < shuiyuan-v1.0.0.tar.gz
```

接收端还会独立检查 GitHub 记录的 asset digest。准备好 shared 下的配置和 secrets 后，生成首次启动用的镜像环境文件；这里没有明文应用密钥：

```bash
sudo python3 - <<'PYCODE'
import json
from pathlib import Path
root = Path('/opt/shuiyuan')
release = json.loads((root / 'releases/v1.0.0/release.json').read_text())
values = {
    'SHUIYUAN_CONFIG': str(root / 'shared/deployment.toml'),
    'SHUIYUAN_SECRETS': str(root / 'shared/secrets'),
    **{name.upper() + '_IMAGE': digest for name, digest in release['images'].items()},
}
path = root / 'shared/initial.env'
path.write_text(''.join(f'{key}={value}\n' for key, value in values.items()))
path.chmod(0o600)
PYCODE
sudo docker compose --env-file /opt/shuiyuan/shared/initial.env --project-name shuiyuan -f /opt/shuiyuan/releases/v1.0.0/deploy/compose.yaml up -d --wait postgres
sudo docker compose --env-file /opt/shuiyuan/shared/initial.env --project-name shuiyuan -f /opt/shuiyuan/releases/v1.0.0/deploy/compose.yaml run --rm migrate
```

在启动 Bot 前，按 deployment.md 执行语料和长期记忆的导入；导入命令同样使用上述 env-file、project-name 和该版本 Compose 路径。重新向量化会调用外部模型，应先 dry-run。不要把首次初始化流程用于绕过已有安装的发布保护。

```bash
sudo docker compose --env-file /opt/shuiyuan/shared/initial.env --project-name shuiyuan -f /opt/shuiyuan/releases/v1.0.0/deploy/compose.yaml up -d --wait bot mcp
```

若配置关闭了 MCP，则启动命令只列 bot。后续控制器会根据配置处理 MCP。

管理员确认实际运行镜像 digest 与该 release.json 一致，`/api/runtime-health` 中 process/database/state/forum 均为 ok 后：

```bash
sudo shuiyuan-release adopt --release v1.0.0
```

登记保存版本与向量空间指纹，建立 current。已有 shared/current.json 时拒绝重复登记，防止跳过迁移兼容检查。首次安装使用的新健康接口要求明确的 SQLite state 字段，旧版本若没有该字段需先按维护流程升级。

## 4. 日常部署与恢复

在 Actions → Deploy production → Run workflow，选择 main，输入已发布版本和 deploy。工作流获取正式 Release，校验部署包，传输到不可变 releases/version，然后执行固定入口。

服务器流程：锁 → 8GiB 磁盘余量和架构检查 → 拉取镜像 → 指纹及已有业务状态检查 → 停 Bot → 使用旧版本工具一致性备份 → 必要的兼容迁移 → 启动单个 Bot → 180 秒业务就绪检查（必须看到本次启动后的新论坛轮询） → 更新 current 与发布记录。

普通发布保持 PostgreSQL digest 不变；不更新数据库服务。不要让额外导入进程或外部写入者在发布期间运行。项目假定当前单实例 Bot/Web 是唯一生产写入者，导入必须由管理员安排在维护窗口。

失败前未停机则保持旧服务。停止后失败，会在兼容规则允许下恢复旧应用，不自动恢复数据库。state/论坛未就绪也视为失败，不以 HTTP 200 代替健康。收到失败时检查：

```bash
sudo shuiyuan-release status
sudo cat /opt/shuiyuan/shared/last-deployment.json
```

记录仅包含版本、阶段、错误类型和备份路径，不写 Cookie 或模型错误原文。管理员可在服务器通过 Compose 查看详细日志，避免将生产日志上传公开 CI artifact。

回滚在同一工作流选择 rollback 和旧版本；必须通过当前版本向该旧版本的兼容判断，且仍先备份。数据库镜像、Embedding 指纹改变、manual 迁移一律拒绝普通回滚。需要恢复数据库时，停止所有写入者，按 deployment.md 在空数据库和空状态卷恢复匹配备份；明确核对备份后的新增记忆及发帖结果，禁止无条件覆盖。

发布并发受 GitHub concurrency 与 Linux flock 双重约束，进行中的部署不被新任务取消。网络断开后先检查 status，不能假设服务器动作已经停止。

## 5. 测试与验收

```bash
uv run --no-sync pytest -q
uv run --no-sync black --check src test scripts deploy/mcp
uv run --no-sync isort --check-only src test scripts deploy/mcp
npm --prefix web ci
npm --prefix web run build
# 镜像已构建时，创建、运行并清理全隔离测试栈：
python3 scripts/ci/integration.py
# 仅在独立的目标规格机器运行：
VALIDATION_SECONDS=86400 python3 scripts/ci/integration.py
```

集成驱动自动生成合成凭据、复用生产角色初始化、测试运行账号无 DDL 权限、非 root 只读根启动、MCP、真实 Bot/Web 入口的模拟 HTTP 连接、登录失败时 Web 可用、重启、备份恢复及资源打包。论坛 HTTP 重定向仅通过测试挂载的 sitecustomize 生效，不进入生产镜像；测试网络无外网出口。

24 小时模式只是合成负载。宿主机同时记录 MemAvailable、OOM、swap、磁盘和重启次数；至少约 200MiB 余量、无 OOM 和持续内存增长才满足容量目标。不能在同一台生产 2GB 服务器并行运行验收栈。真实账号 Cookie 轮换、受控发帖、GitHub/GHCR 远端权限以及真实 SSH 部署需要维护者显式启用并验收。

## 6. 本次实施验证记录

2026-09-08：全量离线 pytest 为 **166 passed、20 skipped**；远程无本地 ML/Neo4j 依赖的独立环境在增加部署控制器测试前完成 **154 passed、20 skipped**。177 个 Python 文件通过 Black/isort，三份工作流通过 actionlint（未启用额外 shellcheck），两套 Compose 静态校验通过。CI 固定 uv 0.8.15 已核验提交中的锁文件与项目声明一致。

三个 linux/amd64 镜像在 Docker Desktop 上完成构建；隔离集成栈通过，包括新增模拟 OpenAI Embedding HTTP 请求、Web 会话创建，以及恢复后的 UI 密钥解密。发行包完成打包、接收、文件白名单及摘要校验的往返检查。检查使用提交中的 uv.lock，而不是用户尚未提交的镜像源变更。

未验证 GitHub 托管工作流的实际运行、GHCR 推送/私有拉取权限、真实 Release 的服务器侧在线核验、真实服务器 SSH 发布、旧正式版本的兼容矩阵和 2GB/24 小时压力测试。未执行推送、合并、Release 发布、生产部署、真实模型计费或社区发帖。
