# ShuiyuanAutoReply

项目采用模块化单体与端口/适配器结构。论坛 Worker 和 HTTP API 共享
`BotService`，渠道负责输入映射与输出发布，应用核心不依赖 FastAPI、
aiohttp 或数据库 SDK。

## 安装与启动

```bash
pip install -e .
shuiyuan-bot                    # 默认 wolf_lumine
shuiyuan-bot 存档读取           # 保留原位置参数语义
```

API 依赖保持可选：

```bash
pip install -e '.[server]'
shuiyuan-api                         # 仅管理站与网页 Runtime
shuiyuan-bot wolf_lumine --web       # Worker + 管理站
shuiyuan-bot wolf_lumine --web --web-host 127.0.0.1 --web-port 11451
```

`shuiyuan-bot` 默认不启动管理站。管理站默认只监听 `127.0.0.1:11451`；
页面无需管理令牌即可直接访问。正式页面由 FastAPI 同源托管，前端源代码位于
`web/`。如将监听地址改为 `0.0.0.0` 或通过反向代理暴露到其他设备，应在代理层
自行增加认证与访问控制。

网页会话默认直接聊天，`【帮助】` 和 `【rua】` 分别进入网页专用 Handler。
网页 Runtime 与论坛 Bot 固定使用 `deepseek-v4-flash-vision-exp` 和
`DEEPSEEK_API_KEY`，不配置 fallback。网页支持一次上传最多 20 张、每张 20MB
的 JPEG、PNG、GIF 或 WebP 图片；论坛私有图片会自动下载并通过 DeepSeek
Files API 送模，网页和论坛搜索结果中的显式图片也会自动进入视觉上下文。
论坛与网页的 Prompt、Session 和长期记忆 namespace 相互隔离。

设置页会主动探测 `MCP_SERVER_URL`，分别为网页和论坛 Runtime 显示连接状态
及工具清单。MCP 工具使用独立的禁用列表，新发现的工具默认启用；可取消勾选
并通过“应用并热切换”使配置生效。

兼容 HTTP 协议仍保留 `POST /api/chat`、`POST /api/clear` 和
`GET /api/health`。环境变量及默认值见 `.env.example`。

## 本地状态与前端构建

默认状态目录为 `~/.shuiyuan-auto-reply`，可用 `SHUIYUAN_STATE_DIR` 覆盖：

```text
state.sqlite3   # 会话、消息、结构化执行轨迹和 Runtime 草稿/版本
master.key      # UI API Key 加密主密钥（0600）
artifacts/      # 用户上传、搜索、论坛与生成图片的会话级本地副本
```

```bash
cd web
npm install
npm run build
```

前端产物写入 `src/shuiyuan_auto_reply/interfaces/api/static` 并随 wheel 打包。

## 测试

```bash
pytest -q
pytest -q --run-live            # 显式启用真实模型、MCP 等外部调用
npm --prefix web run build
```

架构、生命周期和扩展方式见 [架构文档](ARCHITECTURE.md) 与
[扩展指南](EXTENDING.md)。
