# 论坛实时执行监控

论坛记录按话题展示，每个已命中指令的请求拥有独立任务卡片。确认命中后立即记录收到的指令；排队中的指令不会提前加入模型上下文。普通帖子、bot 自回复、未提及 bot 或提及但未命中指令的帖子不创建监控记录。

任务状态为 `queued → running → publishing → completed`，异常状态包括 `failed`、`interrupted` 和 `needs_review`。只有论坛明确确认发布成功才完成；错误提示成功发布仍保留原任务失败状态。发送结果不确定时保留原有队列的人工核实语义，不自动重发。

前置检查最多并发 3 条；执行遵循现有全局并发限制，调度按帖子而非话题加锁，同一话题内先到的请求不阻塞后到的请求。重启会核对持久化队列，结束旧运行记录；安全重试以新 run 关联 `previous_run_id`。队列按帖子 ID 保持顺序，状态更新时间不会改变重试顺序。

## 接口与兼容

- `GET /api/forum/monitor`：同一读事务内返回活动 `runs`、论坛 `conversations`（含排队和执行数量）和事件 `cursor`。
- `GET /api/forum/events/stream?after=N`：只读 SSE，支持 `Last-Event-ID`；每条 `forum.event` 携带 `event_id`、`conversation_id`、`run_id`、`type`、`created_at`、`payload`。每批最多 200 条，空闲时每 500ms 查询、每 15 秒心跳。
- 会话详情增加 `runs`，每项包含 `request` 元信息、状态和 `last_event_id`；列表增加 `queued_count`、`running_count`。旧响应字段保留。

新增接口遵循管理站现有访问边界，不占用回复调度槽位。反向代理须允许长连接并关闭 SSE 缓冲；接口已发送 `X-Accel-Buffering: no`。监控断开不取消 Worker 任务，重连通过新快照和游标补齐；当前话题的完整历史重新读取，事件按 ID 去重。浏览器标签页内记住所选渠道与最近会话，窄屏通过话题选择框切换。

沿用 SQLite 的 runs/run_events 表和 JSON 载荷，不增加数据库字段、服务或迁移要求。旧记录沿用原展示，不清理历史未命中记录。

## 验证与发布

离线验证：

```sh
.venv/bin/python -m pytest -q
npm --prefix web test
npm --prefix web run build
```

Python 的部分既有测试需要启动本地模拟 HTTP 服务；不用 `--run-live`，不会启用真实论坛/模型测试。前端测试需要 Node 22.18 或更新版本，与 CI 使用版本一致。

构建发布镜像：

```sh
docker build --platform linux/amd64 -f deploy/Dockerfile -t shuiyuan-bot:forum-monitor-local .
```

云端使用现有发布流程更新 `bot` 镜像，前端资源已经包含在镜像中；不必重建 PostgreSQL 或 MCP。发布前确认论坛活动任务已清空，停止写入并按现有部署流程备份状态库，保留旧镜像与数据卷。重启后检查 `/api/live`、论坛快照以及一次获准的真实指令；出现问题时停止新版本并切回旧镜像，不删除数据卷。此改造不自动操作云端。


## 本次验收记录（2026-09-09）

- 本机默认离线套件：185 项通过、20 项真实服务测试跳过；前端 3 项测试、类型检查和生产构建通过。
- Linux amd64 镜像 `shuiyuan-bot:forum-monitor-local` 构建通过。无网络、只读根文件系统、临时状态库的容器内，论坛监控与调度测试 17 项通过。
- 浏览器模拟了两个话题、三条请求：观察到指令、实时工具步骤、独立任务卡片及完成结果；检查了桌面侧栏数量、窄屏话题切换和刷新后的会话恢复。
- 没有发送真实论坛消息，没有部署云端。上述验证不替代发布后的真实环境验收。
