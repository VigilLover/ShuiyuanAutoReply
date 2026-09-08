# 首次部署与 GitHub 自动部署操作指南

本文整理自首次部署说明，命令按执行位置排列。示例使用 Ubuntu 24.04 amd64、Bot 账号 `wolf_lumine`、首次版本 `v1.0.0`。将 `SERVER_IP` 和 `ADMIN_USER` 替换为实际服务器 IP、管理员账号。

**先完成一次人工初始化和版本登记，之后再使用 GitHub 手动触发的自动部署。** 当前配置统一从 main 发布并提供部署入口，dev 保留为开发分支。Fork 主线切换见 [分支策略](branch-strategy.md)；必须先把这些修改合入 main，再打正式标签，服务器初始化步骤不变。

执行顺序：发布首个版本 → 准备 Cookie/Key 和数据 → 配置服务器 → 初始化数据库及语料 → 启动并登记 → 配置 GitHub SSH 部署。本文命令是操作说明，不代表已经执行。

## 1. 本机与 GitHub：发布第一个版本

在项目仓库中推送已经提交的实现：

```bash
git push origin dev
```

推送只包含提交，不会包含暂存区或工作区尚未提交的文件；不要为了推送而执行 `git add .`。其他工作区改动应由维护者单独决定是否提交。

在 GitHub 中：

1. 确认完整的部署改造及本次 main 发布配置已提交到 dev。
2. 创建 `dev → main` 的 PR，等待 CI 两套 Python 依赖检查及前端构建通过后合并；完整产品代码和工作流一起进入 main，不再单独复制工作流。
3. 在 Settings → Actions → General 确认允许工作流和官方 Actions 运行，并确认默认分支为 main。
4. 首次 CI 运行后，将 `Python (remote)`、`Python (local)`、`frontend` 设为 main 的必需检查；dev 可同时保留检查。

GitHub 的 workflow_dispatch 入口要求工作流存在于默认分支，当前 Deploy 工作流还明确限制从 main 运行。[GitHub 手动工作流说明](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow?tool=webui)

目标提交合入 main 后创建标签，无需切换本机工作区：

```bash
git fetch origin
git tag v1.0.0 origin/main
git push origin v1.0.0
```

若该标签已存在，使用新的版本号，不移动或覆盖正式标签。

等待 Actions → Release 成功。应看到正式 Release v1.0.0，附件 `shuiyuan-v1.0.0.tar.gz`、`SHA256SUMS`、`release.json`，以及 GHCR 中 bot/postgres/mcp 三个镜像。任务失败时先查看日志，不继续生产初始化。

## 2. 本机：准备 Cookie、Key 和迁移数据

在项目根目录安装本地完整依赖：

```bash
uv sync --locked --extra dev --extra server --extra local-embedding --extra neo4j
```

将自己生成且可信的 Cookie pickle 转为 JSON：

```bash
uv run --no-sync shuiyuan-ops cookie convert \
  cookies /tmp/forum-cookie.json --trust-pickle
```

没有 Cookie 时，在可信本机使用现有 get_cookies.ipynb 完成登录。不得转换来源不明的 pickle；远程只使用 JSON。

初始化秘密文件：

```bash
python3 scripts/deploy/init_secrets.py --cookie /tmp/forum-cookie.json
```

按提示隐藏输入 Embedding Key、DeepSeek Key、图片生成 Key。图片功能暂不使用时可留空，但保留生成的 image_key 文件，当前 Compose 会挂载它。脚本同时生成独立数据库账号密码和 DSN，不打印秘密值。

若提示文件已存在，不要删除后重建，以免改变已在使用的数据库密码。已有部署仅缺 image_key 时可使用脚本的 `--image-key-only`。

仅当配置文件尚不存在时复制示例：

```bash
cp -n config/deployment.example.toml config/deployment.toml
```

编辑 config/deployment.toml：

| 远程配置项 | 内容 |
|---|---|
| embedding.base_url | 百炼实际 OpenAI 兼容 API 地址，地域/业务空间必须与 Key 匹配 |
| embedding.model | 确认账号可用的 qwen3.7-text-embedding |
| embedding.dims | 1024 |
| forum.bot_username | Cookie 对应的账号；本文按 wolf_lumine 配置 |

保留 `/run/secrets/...`、`/var/lib/shuiyuan`、`auto_migrate=false`、pgvector 等容器配置。不能把百炼控制台网页 URL 填成 API 地址。若使用其他人物/账号，还需核对人物配置、启动 persona 与语料人物 ID，不只是替换 Cookie。

需要迁移现有数据时，先停止本地 Bot 及会写长期记忆的 Web Runtime，然后导出：

```bash
mkdir -p transfer
uv run --no-sync shuiyuan-ops --config config/deployment.toml --profile local \
  corpus export transfer/corpus.jsonl --persona wolf_lumine
uv run --no-sync shuiyuan-ops --config config/deployment.toml --profile local \
  memory export transfer/memory.jsonl
```

也可用 CSV 导出替代 corpus 命令：

```bash
uv run --no-sync shuiyuan-ops --config config/deployment.toml --profile local \
  corpus export transfer/corpus.jsonl \
  --csv user_archive/wolf_lumine/user_archive.csv --persona wolf_lumine
```

输出文件已存在时拒绝覆盖。没有历史数据可跳过导出和导入；不要让本地与远程 Bot 同时回复同一账号。

## 3. 服务器：运行环境与网络

选择 Ubuntu Server 24.04 amd64、2 vCPU、2GB RAM、40GB SSD。安装 Python 3 和 Docker Engine/Compose 插件，Docker 使用[官方 Ubuntu 安装步骤](https://docs.docker.com/engine/install/ubuntu/)。

```bash
python3 --version
sudo docker version
sudo docker compose version
sudo docker run --rm hello-world
df -h /
```

预留至少约 8GB 可用磁盘。服务器不需要安装 Node、本地 Embedding 或项目 Python 依赖。

只开放 SSH 22，不公开 11451、5432、58000。服务器需要访问 GitHub/GHCR、模型服务和水源；GitHub Runner 也必须能连到服务器 SSH。若安全组只允许个人 IP，需要另行允许受控 Runner 的网络路径，不能期待自动部署直接连通。

## 4. 本机传输；服务器安装控制器与配置

从 GitHub 正式 Release 下载部署包和 SHA256SUMS，在本机执行：

```bash
ssh ADMIN_USER@SERVER_IP 'mkdir -p ~/shuiyuan-bootstrap'
scp shuiyuan-v1.0.0.tar.gz SHA256SUMS ADMIN_USER@SERVER_IP:~/shuiyuan-bootstrap/
scp config/deployment.toml ADMIN_USER@SERVER_IP:~/shuiyuan-bootstrap/
scp -r secrets transfer ADMIN_USER@SERVER_IP:~/shuiyuan-bootstrap/
```

没有 transfer 时从最后一条命令中去掉它。登录服务器后：

```bash
ssh ADMIN_USER@SERVER_IP
cd ~/shuiyuan-bootstrap
sha256sum -c SHA256SUMS
mkdir unpacked
tar -xzf shuiyuan-v1.0.0.tar.gz -C unpacked
sudo python3 unpacked/scripts/deploy/install_controller.py
sudo install -m 600 deployment.toml /opt/shuiyuan/shared/deployment.toml
sudo cp secrets/* /opt/shuiyuan/shared/secrets/
sudo chown -R root:root /opt/shuiyuan/shared/secrets
sudo chmod 700 /opt/shuiyuan/shared/secrets
sudo find /opt/shuiyuan/shared/secrets -type f -exec chmod 444 {} \;
```

这些复制命令只用于新安装；既有配置和秘密文件轮换应按运维指南操作。私有目录内的秘密文件通过单文件 bind mount 提供给容器，Compose secrets 不提供宿主机静态加密。

## 5. 服务器：GitHub/GHCR 只读访问与接收版本

私有 GHCR 镜像需要 Personal access token（classic），授予 read:packages，并确保账号能读取对应 package：

```bash
sudo docker login ghcr.io -u VigilLover
```

在 Password 提示中输入 Token，不放进命令参数。使用 sudo 是因为控制器以后读取 root 的 Docker 登录配置。[GHCR 认证说明](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)

私有代码仓库还需独立的 Fine-grained token，仅授权本仓库 Contents: Read-only：

```bash
sudo install -m 600 /dev/null /opt/shuiyuan/shared/github_read_token
sudo nano /opt/shuiyuan/shared/github_read_token
```

文件只放 Token 本身；公开代码仓库可省略，但仍受未认证 API 限流影响。

接收版本：

```bash
cd ~/shuiyuan-bootstrap
bundle_sha=$(sha256sum shuiyuan-v1.0.0.tar.gz | cut -d ' ' -f 1)
sudo env SSH_ORIGINAL_COMMAND="receive v1.0.0 $bundle_sha" \
  /usr/local/sbin/shuiyuan-ssh < shuiyuan-v1.0.0.tar.gz
```

成功会返回 received: v1.0.0。服务器会独立核对 GitHub 正式 Release 的附件摘要，API 不可用或摘要不匹配时停止。

## 6. 服务器：初始化数据库

生成首次版本环境文件：

```bash
sudo python3 - <<'PY'
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
PY
```

在当前服务器终端定义函数，避免每次重复长参数：

```bash
dc() {
  sudo docker compose \
    --env-file /opt/shuiyuan/shared/initial.env \
    --project-name shuiyuan \
    -f /opt/shuiyuan/releases/v1.0.0/deploy/compose.yaml "$@"
}
dc config --quiet
dc pull postgres bot mcp
dc up -d --wait postgres
dc run --rm migrate
```

dc 只对应首次版本 v1.0.0，重新登录需要重新定义。后续版本通过控制器管理，不能一直用该函数更新生产。现在只初始化数据库、SQLite 和队列，尚未启动 Bot。

## 7. 服务器：导入语料和长期记忆

有迁移数据时：

```bash
sudo cp -R ~/shuiyuan-bootstrap/transfer /opt/shuiyuan/transfer
sudo chown -R 10001:10001 /opt/shuiyuan/transfer
sudo chmod 700 /opt/shuiyuan/transfer

dc run --rm -v /opt/shuiyuan/transfer:/transfer bot \
  shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote \
  corpus import /transfer/corpus.jsonl --dry-run

dc run --rm -v /opt/shuiyuan/transfer:/transfer bot \
  shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote \
  memory import /transfer/memory.jsonl --dry-run
```

通过后正式导入。**以下操作把文本发送给外部 Embedding 服务并产生费用：**

```bash
dc run --rm -v /opt/shuiyuan/transfer:/transfer bot \
  shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote \
  corpus import /transfer/corpus.jsonl

dc run --rm -v /opt/shuiyuan/transfer:/transfer bot \
  shuiyuan-ops --config /etc/shuiyuan/deployment.toml --profile remote \
  memory import /transfer/memory.jsonl
```

中断后执行相同命令继续，不删除断点。没有某类数据就跳过对应命令。

## 8. 服务器：预检、启动和登记

```bash
dc run --rm bot shuiyuan-ops \
  --config /etc/shuiyuan/deployment.toml --profile remote config check
dc run --rm bot shuiyuan-ops \
  --config /etc/shuiyuan/deployment.toml --profile remote doctor --probe-forum
```

需要单独检查 Embedding 时显式运行以下付费探测：

```bash
dc run --rm bot shuiyuan-ops \
  --config /etc/shuiyuan/deployment.toml --profile remote doctor --probe-embedding
```

确认本地旧 Bot 已停止，再启动：

```bash
dc up -d --wait bot mcp
dc logs --tail=100 bot
curl -s http://127.0.0.1:11451/api/runtime-health
```

process、database、state、forum 四个字段均应为 ok；forum 暂未就绪时等待几十秒并检查日志。不能仅凭容器 healthy 判定社区正常。若配置关闭 MCP，启动命令只列 bot。

正常后登记首个版本：

```bash
sudo shuiyuan-release adopt --release v1.0.0
sudo cat /opt/shuiyuan/shared/current.json
```

**adopt 是后续自动部署的前提。** 管理员应先确认当前容器镜像与该版本 release.json 一致。

本机建立管理页面隧道：

```bash
ssh -N -L 11451:127.0.0.1:11451 ADMIN_USER@SERVER_IP
```

访问 http://127.0.0.1:11451，检查聊天模型、工具和 Key。最后在社区进行一条受控提及验收，确认没有重复回复。真实发帖由部署者显式执行。

## 9. 本机、服务器与 GitHub：配置受限部署 SSH

本机创建专用密钥，不复用管理员密钥：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/shuiyuan-ci-deploy -C shuiyuan-ci-deploy -N ""
```

服务器上创建专用账号并安装公钥：

```bash
sudo useradd --create-home --shell /bin/bash shuiyuan-deploy
sudo chown root:root /home/shuiyuan-deploy
sudo chmod 755 /home/shuiyuan-deploy
sudo install -d -o root -g root -m 755 /home/shuiyuan-deploy/.ssh
sudo nano /home/shuiyuan-deploy/.ssh/authorized_keys
```

authorized_keys 写入以下前缀和本机 .pub 的完整公钥：

```text
restrict,command="sudo -n /usr/local/sbin/shuiyuan-ssh" ssh-ed25519 AAAA... shuiyuan-ci-deploy
```

```bash
sudo chown root:root /home/shuiyuan-deploy/.ssh/authorized_keys
sudo chmod 644 /home/shuiyuan-deploy/.ssh/authorized_keys
sudo visudo -f /etc/sudoers.d/shuiyuan-deploy
```

sudoers 内容：

```text
Defaults:shuiyuan-deploy env_keep += "SSH_ORIGINAL_COMMAND"
shuiyuan-deploy ALL=(root) NOPASSWD: /usr/local/sbin/shuiyuan-ssh ""
```

```bash
sudo visudo -c
sudo cat /etc/ssh/ssh_host_ed25519_key.pub
```

不要把部署账号加入 Docker 组。主机公钥通过可信服务器终端获取，而不是在 CI 里临时扫描并盲目信任。

GitHub Settings → Environments 创建 production，添加：

| Secret | 内容 |
|---|---|
| DEPLOY_HOST | IP 或域名，不含 ssh://；当前只支持 SSH 22 |
| DEPLOY_USER | shuiyuan-deploy |
| DEPLOY_SSH_KEY | 本机专用私钥完整内容 |
| DEPLOY_KNOWN_HOSTS | `SERVER_IP ssh-ed25519 AAAA...`，公钥来自可信服务器终端 |

KNOWN_HOSTS 的主机名必须与 DEPLOY_HOST 一致。不要误放服务器私钥。

在本机核验主机身份后测试：

```bash
ssh -i ~/.ssh/shuiyuan-ci-deploy -o IdentitiesOnly=yes shuiyuan-deploy@SERVER_IP status
```

仅执行过 adopt 时返回空 JSON 可以正常。普通 shell 命令被拒绝属于预期行为。

## 10. 后续更新和失败处理

修改代码 → PR 合入 main → CI 成功 → 从 origin/main 打新标签（可先在 dev 集成）：

```bash
git fetch origin
git tag v1.0.1 origin/main
git push origin v1.0.1
```

Release 会检查标签对应的提交是否属于 origin/main 的历史；仅存在于 dev 的新提交不能直接发布。

Release 成功后，Actions → Deploy production → Run workflow，选择 main，version 填 v1.0.1，operation 选 deploy。等待完成后：

```bash
sudo shuiyuan-release status
curl -s http://127.0.0.1:11451/api/runtime-health
```

自动化执行拉取、停机备份、允许的迁移、启动及新论坛轮询检查。**不会每次推送就自动更新生产，部署仍由你手动触发。**

涉及数据库镜像、Embedding 或 schema 变化时，按 [CI/CD 运维指南](cicd.md) 处理兼容性，不能关闭保护硬部署。失败时先查 status 和日志，不重复恢复数据库；应用回滚不等于数据库恢复。

当前已完成本地离线及隔离 Linux 容器验证，但 GitHub 实际发布、真实服务器 SSH 与 2GB/24 小时容量测试仍需要部署环境验收。
