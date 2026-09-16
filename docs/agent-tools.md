# Agent 资料与参考素材

论坛与网页共用通用工具；网页的论坛写权限保持关闭。此次更新不新增数据库表，本轮证据、查询缓存、素材集在请求结束后释放，不是长期记忆。

## 帖子和用户

- `get_post(topic_id=None, post_number=None, post_id=None, refresh=False, cursor=0)` 读单帖：用话题内楼层号（`topic_id` + `post_number`）或全站帖子 ID（`post_id`）定位，两种编号不能混用。缺两者之一时返回明确错误。
- 精准读取优先获取 `raw`，缺失时补取详情，再回退为清理后的 HTML 文本。结果包含 `content_source`、回复关系、正文／引用提及、图片、来源和警告。
- 精准正文每页 12000 字符，`truncated`、`next_cursor` 明确指示后续内容。用同一精准工具的 `cursor`，或 `read_tool_result(result_id, cursor)` 读取后续页。
- 搜索正文摘要最多 800 字符，并说明返回结果不保证穷尽。标题可为空；不为标题增加整话题请求。
- `get_user` 精确查用户名，输出保留逐项错误。只在需要头像时传 `include_avatar=True`。多个已知用户逐个调用。
- `search_user(term=None, user_id=None)` 负责模糊发现或按 user_id 反查；查不到不代表用户不存在（user_id 路径依赖用户发帖记录）。
- 相同成功查询仅在当前轮复用；`refresh=True` 请求新数据。批量重试不会重复请求成功用户。非重试性 HTTP 错误不会盲目重试。

## 上下文和工具配置

`common.runtime.context_token_budget` 默认 24000，必须为正值；它是动态文本输入的近似 token 预算，不是模型的真实总窗口大小。工具 schema、静态规则和提供商多模态配额独立处理。当前指令、调用／返回配对等必要信息可能使极端输入超过估算预算。

压缩只修改发送给模型的视图：完整结果在本轮内可回读，目标帖引用、最近两组调用和实体映射仍可访问。不设置按任务类型的搜索次数限制。

每次调用模型前都会校验调用／返回配对：同一轮内重复读取命中缓存时，回放的返回使用新的消息 ID，否则图状态按 ID 合并会顶掉较早的那条消息，使一次调用失去返回；出站前若仍存在没有返回的调用或先于调用出现的返回，会补齐或丢弃并记录 `tool.pairing_repaired`。OpenAI 兼容的 Responses 端点遇到这种输入会直接返回 400，失败的是整轮回答而不只是一次调用。

MCP 的 `get_hardware_status` 与对话和检索无关，新配置默认关闭，可在设置页按应用打开。当前时间由执行控制消息直接给出，模型不必为此调用 `get_system_time`。

默认工具目录包含新增工具。已有显式 `enabled_tools` 白名单不会自动扩大；需要使用新能力时，在管理界面的论坛／网页工具设置中启用 `get_user`、`read_tool_result`、`prepare_image_references`。若未启用 `read_tool_result`，运行时保留完整工具消息，避免产生不可回读的压缩内容；精准读帖仍可用 `cursor` 翻页。

同一能力只暴露一个工具入口：按 ID 查人并入 `search_user`，按全局 `post_id` 读帖并入 `get_post`，近义词工具会让模型在同一个读操作上换名字重试。工具 schema 里声明的 `limit` 一律为 1–20（默认 10），超出范围返回带说明的错误；不支持分页的工具不再暴露 `page` 参数。

## 外部网页读取

`web_search` 和 `web_read` 都通过 MCP；Bot 不再维护第二份网页抓取实现。`web_search` 会先解包 MCP ContentBlock，再恢复 `title`、`url` 和 `snippet`。`web_read` 同时兼容旧版 MCP 的纯文本返回和新版结构化 envelope。

`web_read(url, mode="auto", query=None, json_path=None, fields=None, max_results=20)` 是唯一网页读取入口：

- `auto` 根据 Content-Type 和可解析性选择文档、JSON 或普通文本；`document` 提取 HTML 主体并去除导航、页眉页脚、侧栏、表单和隐藏节点；`raw` 只在确需原文时使用。
- `query` 对文档返回命中块及相邻上下文，对 JSON 集合过滤包含关键词的对象。大型接口优先组合 `json_path` 和 `fields`，避免图片 URL、SKU、库存明细等无关字段占满上下文。
- 默认每页 6000 字符，最多 12000。分页以 MCP 清洗后的完整表示为基准；`next_cursor` 是绑定 URL、模式、查询和字段投影的不透明游标。继续翻页时可以重复传入相同 URL，但不能改变提取条件。
- 每页证据包含 `ref`、`url`、`content` 和 `page_start`。同 URL 的不同页按页偏移和内容分别计入进展，最终生成阶段会保留成功读取的网页证据。

该能力仍是公网只读 HTTP(S)，不提供 Bash、认证请求头、Cookie、任意 HTTP 方法或浏览器脚本执行。

## 参考图片

`prepare_image_references(references)` 接受 1–50 项，每项包含唯一 `key`、`url` 和可选 `label`。返回本轮 `reference_set_id`、成功顺序、逐项错误。成功下载会复用；超时、连接失败、429、5xx 最多尝试 3 次，遵循 Retry-After 和本轮剩余时间。已确认用户的头像 404 时可刷新一次资料，只有 URL 改变后才重试。

`generate_image(..., reference_set_id=...)` 用成功素材建立实际编号，prompt 使用标签描述对象。部分失败时只纳入成功素材对应对象，并在最终回复说明未纳入项；全部失败不会改成无参考生成。旧 `reference_images` URL 列表继续支持，但部分失败会先返回准备结果，要求模型调整描述后再生成，防止旧编号错位。

素材集仅在当前轮有效。新一轮修改旧图时需要重新准备。图像生成成功不表示内容已通过身份、人数或细节核验。

## 验证与观察

默认离线测试包含模拟 HTTP 服务、中文 URL 编码、签名查询串保留、部分失败、编号与图片字节顺序、超长正文、216 条历史压缩、并发轮隔离以及真实 LangGraph 编排测试。运行 `uv run pytest -q`；仅已安装环境验证可使用 `uv run --no-sync pytest -q`。

观察 `context.evidence` 的结果数及缓存命中数、`image.references_prepared` 的成功／失败计数，以及上下文投影的估算 token 日志。线上验证应另外安排，不在离线测试中调用论坛或付费生图服务。

## 托管提示词与迁移（规则版本 4）

运行配置支持 `managed` 和 `legacy`。托管模式每次构建 Runtime 都组合当前代码规则、独立人设和补充要求；人设中的花括号按普通文本处理。旧完整 `system_prompt` 仍保留，供审阅和旧版本读取。已有完整自定义提示词保持 legacy，不猜测拆分内容。

系统用 `prompts/legacy_defaults.json` 的历史精确 SHA256 和当前默认模板识别可自动迁移的配置。active 与 draft 分别迁移，revision 不因启动或读取而增加。迁移写入使用比较并交换，避免覆盖同时保存的草稿。

管理页显示草稿差异、规则版本、配置 revision 和最近任务实际使用的规则。`runtime.profile_used` 记录每次实际请求对应的代码版本/指纹、规则版本、profile revision 和模板 hash。`SHUIYUAN_RELEASE_VERSION` 是可选诊断标签；未设置时显示安装包版本，代码 hash 用于区分实际实现。

新增接口：

- `POST /api/settings/profiles/{scope}/prompt-preview`：传入草稿字段，只读返回组合后的模板和指纹，不保存配置。
- `POST /api/settings/profiles/{scope}/prompt-migrate`：仅把提示词草稿转换为 managed，保留旧全文、模型、API 格式和工具开关；需要审阅后应用。
- `POST /api/settings/profiles/{scope}/restore-persona`：仅恢复默认人设草稿。

“应用并热切换”才使草稿成为 active。论坛 Worker 在后续任务开始时采用新配置；已开始的任务保留原来的快照。网页和论坛独立应用。部署后不需要手动覆盖已托管的执行规则，但 legacy 自定义配置仍需人工审阅迁移。

## 调查进度与收敛

`update_task_progress` 是不能通过普通工具白名单关闭的内部状态工具，不读取外部数据。记录目标、已确认作者、缺口 ID、带证据 ID 的结论和新策略。作者需存在于本轮证据中；结论必须引用存在的证据，不能用更新状态清零执行预算。

论坛搜索支持 `gap_id`、`scope_reason`、`limit`（1–20）。当仅有一个未解决缺口时，控制器可自动关联该缺口；MCP 工具无需修改其服务端 schema。当前话题默认补齐 topic_id。作者过滤严格保持模型传入的取值，控制器不再自动补齐 username：静默改写会让模型拿到自己没有要求的结果。多作者集合需要逐作者查询或说明扩大范围的理由。新的精准定位需要来源、真实回复关系或对应缺口的明确说明。

默认控制参数位于 `[common.runtime]`：

| 配置 | 默认值 |
|---|---:|
| `no_progress_batches` | 3 |
| `continuation_limit` | 1 |
| `continuation_batch_limit` | 2 |
| `query_limit` | 40 |
| `model_limit` | 24 |
| `final_reserve_seconds` | 60 |

`query_limit` 计数外部只读工具调度，批量查询算一次调度；底层请求和缓存命中另行观测，不能将工具次数当成 HTTP 次数。素材准备与图片生成不消耗论坛检索配额，但仍受模型轮次、总时限和媒体预算约束。`read_tool_result` 不消耗外部查询次数，其重复读取仍参与无进展检测。

连续无进展或重复读取会触发复盘；必须提供未解决缺口和不同新策略才能有限续查。续查最多两个读取批次，随后根据已有资料回答。到达预算边界时最终调用不绑定工具。模型无视停止要求、返回空内容或失败时，输出明确的未完成说明，不重启循环。

进展判断基于来源身份、内容 hash、作者/话题范围和未读正文页，不能保证所有新增正文都在语义上有用；总预算提供最终边界。来源引用验证也不等于自动证明模型的每条推论正确，仍需行为评测。

证据索引只收录结构化来源，不再收录索引、压缩包装和执行日志。`read_tool_result` 接受 result_id 或 evidence_id，可通过 `field` 读取单一顶层字段；默认页长 1500，可用 `limit` 调整至最多 12000 字符。控制器会按同批回读数量缩小实际页长；返回 `page_limit` 和 `next_cursor`，不静默截断。无效页和索引回读不算新增证据。当前指令、目标帖和结构化进度在压缩时保留。

## 显式查看图片

普通论坛读帖和文字搜索只返回图片元信息；新增 `inspect_images(urls=None, evidence_ids=None, description="")`，按需选择一至四张图。当前用户直接附带图片仍自动加载；目标帖图片不再自动下载。显式 `image_search` 保留展示素材能力。合照使用 `get_user` → `prepare_image_references` → `generate_image`，不要求逐张识图。

`inspect_images` 属于内置可选工具；已有白名单需要手动启用，管理页会提示缺失能力。原图、secure-uploads 和头像走论坛认证路径，失败后不降级成无鉴权公开下载。成功字节和确定性失败在本轮共享；查看图片与准备参考图复用已下载字节。无效图片、读取失败均不能作为已理解画面的依据。

## 验证与发布检查

离线回归覆盖迁移幂等性、草稿隔离、纯文本花括号、查询范围不被改写、同批状态更新及搜索、换关键词但相同结果、无视复盘指令、有限续查、确定性错误、调用／返回配对以及按需图片路径。

脱敏评测集位于 `test/fixtures/agent_convergence/scenarios.json`，包含音乐偏好、主楼改写、名单合照和必要广泛研究。以下命令默认只显示样例，不调用模型：

```bash
uv run --no-sync python scripts/eval_agent_convergence.py
```

需要单独评测真实模型决策时，显式开启文本模型调用：

```bash
uv run --no-sync python scripts/eval_agent_convergence.py --live-model --scenario music
```

此入口使用 `DEEPSEEK_API_KEY`。论坛、参考图和图片生成工具均为脱敏模拟器，不访问真实论坛或生成真实图片；输出包含答案、人工验收要求和调用指标。它验证真实模型的决策与控制器配合，不代替线上多模态验收。默认离线 pytest 不运行这些付费调用。

新增事件 `tool.execution` 记录补齐范围后的实际参数；`retrieval.batch`、`retrieval.progress`、`retrieval.finished` 记录新增证据、复盘、重复、轮次、查询、缓存、媒体及停止原因；`model.failed` 记录模型调用失败的阶段与错误正文（模型提供商返回的 4xx 正文会一并记录）；`tool.pairing_repaired` 记录出站前修补的调用 ID。`forum_http_requests` 单独统计论坛 GET，包括补取和重试。排查时先确认 `runtime.profile_used`，再分析调用轨迹。

沿用 `backward-compatible / recent:2` 发布策略，没有新增表。发布前仍须运行 Release 的旧镜像兼容性检查；本地 JSON 配置回归不能替代旧版本容器测试。回滚后旧程序读取归档的完整提示词，新模式的人设与补充要求不会自动转换成旧格式。发布、部署和线上发帖验收单独执行。
