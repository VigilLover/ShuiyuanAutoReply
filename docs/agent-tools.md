# Agent 资料与参考素材

论坛与网页共用通用工具；网页的论坛写权限保持关闭。此次更新不新增数据库表，本轮证据、查询缓存、素材集在请求结束后释放，不是长期记忆。

## 帖子和用户

- `forum_read(post_id=None, topic_id=None, post_number=None, ..., images="none")` 读取单帖或话题窗口：用话题内楼层号（`topic_id` + `post_number`）或全站帖子 ID（`post_id`）定位。图片默认不下载，只有显式传入 `images="auto"` 或 `images="selected"` 时才加载。
- 精准读取优先获取 `raw`，缺失时补取详情，再回退为清理后的 HTML 文本。结果包含 `content_source`、回复关系、正文／引用提及、图片、来源和警告。
- 精准正文每页 8000 字符，话题窗口每帖 1400 字符，`next_cursor` 明确指示后续内容；继续读取时使用同一工具的 `cursor`。每帖附 `reply_to`、`reply_to_author` 与 `replies`。
- 搜索摘要最多 400 字符，并说明返回结果不保证穷尽。标题可为空；不为标题增加整话题请求。
- 工具描述统一为中文三段式（何时用／参数要点／返回结构）；错误信封统一为 `status/code/message/retryable`，常见错误附 `hint` 指示模型下一步该换什么调用。
- 同一批调用里的多个 `users(username=…)` 会自动合并成一次 `usernames` 查询，再按 call_id 拆回；`image_refs` 的 `forum:` 前缀与顺序差异不再造成重复读取。
- `users(query=None, username=None, usernames=None, user_id=None, include_avatar=False)` 统一处理精确用户名、批量用户名、模糊发现和 user_id 反查；查不到不代表用户不存在（user_id 路径依赖用户发帖记录）。只在需要头像时启用 `include_avatar`。
- 相同成功查询仅在当前轮复用；批量重试不会重复请求成功用户。非重试性 HTTP 错误不会盲目重试。

## 上下文和工具配置

`common.runtime.context_token_budget` 默认 60000，必须为正值；它是工具循环消息的近似 token 预算，不是模型的真实总窗口大小。对话历史固定 4000 token、近期讨论固定 6000 字符，其余全部留给工具循环。

超预算时整组丢弃最旧的工具轮（保留当前请求、目标帖与最近三组调用），被丢弃的完整结果仍可在本轮回读并进入收尾证据；保留下来的消息逐字节不变，因此每轮请求与上一轮共享前缀，DeepSeek 磁盘缓存可持续命中。每轮变化的执行控制文本追加在提示词末尾，时间只精确到小时。

时间模型分三层：整轮 `timeout`（默认 900s）、单次模型请求 `model_call_timeout`（默认 180s）、收尾保留 `final_reserve_seconds`（默认 150s）。调查轮剩余时间不足收尾保留时立即转入收尾；收尾轮使用独立的 `DEEPSEEK_MENTION_FINAL_REASONING_EFFORT`。默认 `model_limit=14`、`query_limit=30`、`no_progress_batches=2`。

每次调用模型前都会校验调用／返回配对：同一轮内重复读取命中缓存时，回放的返回使用新的消息 ID，否则图状态按 ID 合并会顶掉较早的那条消息，使一次调用失去返回；出站前若仍存在没有返回的调用或先于调用出现的返回，会补齐或丢弃并记录 `tool.pairing_repaired`。OpenAI 兼容的 Responses 端点遇到这种输入会直接返回 400，失败的是整轮回答而不只是一次调用。

当前时间由执行控制消息直接给出，模型不必为此调用 `get_system_time`。SimpleMCP 已移除硬件状态与旧网页读取工具。

默认工具目录包含论坛读取、用户读取和参考素材准备等工具。已有显式 `enabled_tools` 白名单不会自动扩大；需要使用新能力时，应在管理界面的论坛／网页工具设置中启用对应工具。精准读帖和网页读取均可使用各自的 `cursor` 翻页。

同一能力只暴露一个工具入口：用户查询统一为 `users`，论坛读取统一为 `forum_read`。工具 schema 里声明的论坛 `limit` 为 1–20（默认 10），超出范围返回带说明的错误；不支持分页的工具不暴露 `page` 参数。

## 外部网页读取

`web_search` 和 `web_read` 都通过 MCP；Bot 不再维护第二份网页抓取实现。`web_search` 会先解包 MCP ContentBlock，再恢复 `title`、`url` 和 `snippet`。`web_read` 同时兼容旧版 MCP 的纯文本返回和新版结构化 envelope。

`web_read(url, mode="auto", query=None, json_path=None, fields=None, max_results=20, images="none")` 是唯一网页读取入口。图片默认不加载，直接图片 URL 或页面图片只有在显式传入 `images="auto"` 时才进入视觉上下文：

- `auto` 根据 Content-Type 和可解析性选择文档、JSON 或普通文本；`document` 提取 HTML 主体（article/main/body）渲染为轻量 Markdown（标题、列表、表格行、代码块、引用），去除导航、页眉页脚、侧栏、表单、Cookie／订阅弹层、评论区和隐藏节点，并返回页面 `title` 与 `published_at`（若页面声明）；`raw` 只在确需原文时使用。
- 水源社区域名的 URL 会被 `web_read` 直接拒绝并提示改用 `forum_read`，避免无认证抓取。
- `query` 对文档返回命中块及相邻上下文，对 JSON 集合过滤包含关键词的对象。大型接口优先组合 `json_path` 和 `fields`，避免图片 URL、SKU、库存明细等无关字段占满上下文。
- 默认每页 8000 字符，最多 12000。`query` 命中块前后各保留两块上下文。分页以 MCP 清洗后的完整表示为基准；`next_cursor` 是绑定 URL、模式、查询和字段投影的不透明游标。继续翻页时可以重复传入相同 URL，但不能改变提取条件。
- 每页证据包含 `ref`、`url`、`content` 和 `page_start`。同 URL 的不同页按页偏移和内容分别计入进展，最终生成阶段会保留成功读取的网页证据。

该能力仍是公网只读 HTTP(S)，不提供 Bash、认证请求头、Cookie、任意 HTTP 方法或浏览器脚本执行。

`get_chuangka_menu(location="all", category="all", query=None)` 通过 MCP 读取交图、交环创咖的当前菜单。`location` 可选 `all`、`zhutu`、`huanyuan`；`category` 可选完整菜单 `all` 或冰淇淋菜单 `ice_cream`；`query` 仅按商品名过滤。输出正文只保留商品名和价格。长菜单通过绑定筛选条件的不透明 `cursor` 继续读取，并作为完整网页证据进入最终生成阶段。

## 参考图片与生成结果

`generate_image(prompt, aspect_ratio="1:1", references=None, allow_partial=False)` 是唯一生图入口。`references` 为 `[{"key","url","label"}]`，程序内部完成下载、校验、去重与编号：成功下载会复用；超时、连接失败、429、5xx 最多尝试 3 次，遵循 Retry-After 和本轮剩余时间；已确认用户的头像 404 时刷新一次资料。任一素材失败时默认不生成并返回 `{"status":"partial","failed":[…],"loaded":[…],"hint":…}`；只有 `allow_partial=true` 才用成功子集生成。

工具返回结构化 JSON：成功为 `{"status":"ok","artifact":"artifact://…","width","height"}`，模型必须用 `![描述](artifact://…)` 嵌入最终回复；失败为 `{"status":"error","code","message"}`。Responses API 路径下生成结果会以 512px 低精度预览附回 `function_call_output`，模型能看到自己生成的图再决定是否重画。生成并发由 `common.runtime.image_concurrency`（默认 2）控制。图像生成成功不表示内容已通过身份、人数或细节核验。

## 验证与观察

默认离线测试包含模拟 HTTP 服务、中文 URL 编码、签名查询串保留、部分失败、编号与图片字节顺序、超长正文、216 条历史压缩、并发轮隔离以及真实 LangGraph 编排测试。运行 `uv run pytest -q`；仅已安装环境验证可使用 `uv run --no-sync pytest -q`。

观察 `context.evidence` 的结果数及缓存命中数、`image.references_prepared` 的成功／失败计数，以及上下文投影的估算 token 日志。线上验证应另外安排，不在离线测试中调用论坛或付费生图服务。

## 托管提示词与迁移（规则版本 6）

规则版本 6 把论坛／网页系统模板压缩到约 5KB：合并重复的「不编造」「不复述过程」条目，工具章节改为「先想缺什么，再用最少调用补齐；同一资料只读一次；多个用户一次查」，删除已失效的 `prepare_image_references`、`get_user`、`reference_set_id` 等说法；`legacy_defaults.json` 收录了 v4／v5 模板指纹，旧默认提示词首次读取时自动识别为 managed 并显示 `migration_required`。运行配置支持 `managed` 和 `legacy`。托管模式每次构建 Runtime 都组合当前代码规则、独立人设和补充要求；人设中的花括号按普通文本处理。旧完整 `system_prompt` 仍保留，供审阅和旧版本读取。已有完整自定义提示词保持 legacy，不猜测拆分内容。

系统用 `prompts/legacy_defaults.json` 的历史精确 SHA256 和当前默认模板识别可自动迁移的配置。active 与 draft 分别迁移，revision 不因启动或读取而增加。迁移写入使用比较并交换，避免覆盖同时保存的草稿。

管理页显示草稿差异、规则版本、配置 revision 和最近任务实际使用的规则。`runtime.profile_used` 记录每次实际请求对应的代码版本/指纹、规则版本、profile revision 和模板 hash。`SHUIYUAN_RELEASE_VERSION` 是可选诊断标签；未设置时显示安装包版本，代码 hash 用于区分实际实现。

新增接口：

- `POST /api/settings/profiles/{scope}/prompt-preview`：传入草稿字段，只读返回组合后的模板和指纹，不保存配置。
- `POST /api/settings/profiles/{scope}/prompt-migrate`：仅把提示词草稿转换为 managed，保留旧全文、模型、API 格式和工具开关；需要审阅后应用。
- `POST /api/settings/profiles/{scope}/restore-persona`：仅恢复默认人设草稿。

“应用并热切换”才使草稿成为 active。论坛 Worker 在后续任务开始时采用新配置；已开始的任务保留原来的快照。网页和论坛独立应用。部署后不需要手动覆盖已托管的执行规则，但 legacy 自定义配置仍需人工审阅迁移。

## 调查进度与收敛

`TaskProgress` 只记录本轮目标、话题、已执行搜索和当前阶段；`RetrievalControl` 根据新增来源、重复结果、查询次数、模型次数和剩余时限决定继续调查或进入最终生成。证据索引用于给模型提供成功取得的结构化资料，不承担逐句引用校验、可靠性评分或“完整结论”认证。

论坛搜索保持模型传入的作者和话题范围，不静默扩大查询。当前话题可由运行时补齐 `topic_id`；精准定位仍应来自当前请求、真实回复关系或已取得的资料。

默认控制参数位于 `[common.runtime]`：

| 配置 | 默认值 |
|---|---:|
| `no_progress_batches` | 3 |
| `query_limit` | 40 |
| `model_limit` | 24 |
| `final_reserve_seconds` | 60 |

`query_limit` 计数外部只读工具调度，批量查询算一次调度；底层请求和缓存命中另行观测，不能将工具次数当成 HTTP 次数。素材准备与图片生成不消耗论坛检索配额，但仍受模型轮次、总时限和媒体预算约束。同一工具的游标续读仍参与无进展检测。

连续无进展、明确完成或到达预算边界时进入不绑定工具的最终调用。调查阶段模型调用失败时允许一次纯文本恢复调用；最终调用仍失败，或纯文本修复后正文仍为空／仍含工具标记，才认定整轮完全失败。此时论坛发布 `抱歉，发生了未知错误` 并将任务标记为 `failed`，网页 API 返回同文案的 HTTP 500；失败回复不写入对话历史，也不产生 `model.completed`。

进展判断基于来源身份、内容 hash、作者／话题范围和未读正文页，不能保证所有新增正文都在语义上有用；总预算只提供执行边界。系统保留禁止编造帖子、用户、图片内容和图片链接的约束，但不会强制最终正文附带证据注释或可靠性声明。

证据索引只收录成功取得的结构化来源，不收录工具失败、重试记录、压缩包装和执行日志。最终生成上下文只包含当前请求、对话上下文和成功资料；当前指令、目标帖和结构化进度在压缩时保留。最终正文不描述查询、工具、失败、重试或核实过程，用户未要求时也不主动添加引用、注释或可靠性声明；非关键资料缺失时直接忽略。

## 按需查看图片

`forum_read` 和 `web_read` 默认使用 `images="none"`，普通读帖、网页读取和文字搜索只返回图片元信息。模型只有在确需看图时才显式传入 `images="auto"`；`forum_read` 还支持 `images="selected"` 和 `image_refs`，按引用选择图片。当前请求直接附带的图片仍自动加载；被回复楼层和普通工具结果中的图片不自动下载。显式 `image_search` 保留展示素材能力。合照使用 `users(usernames=[…], include_avatar=true)` → `generate_image(references=[…])`，不要求逐张识图。

原图、secure-uploads 和头像走论坛认证路径，失败后不降级成无鉴权公开下载；`secure-uploads/original/…/<sha1>.<ext>` 与 `optimized/…_2_WxH.<ext>` 会先换算成 `upload://<base62(sha1)>.<ext>` 短地址再下载（sha1 直链需要签名，Bot 拿不到）。长边 ≤512px 且 ≤300KB 的图片以 `detail=low` 内联发送，其余经 Files API 上传并缓存 file_id 七天；只有提供商返回 400 才把内联图改走 Files API。成功字节和确定性失败在本轮共享；查看图片与准备参考图复用已下载字节。下载、上传或解析失败只记入 `image_failures`、运行事件和管理日志，不加入最终正文；图片不可用不算整轮失败，模型继续依据现有文字回答，也不据此猜测图片内容。

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

新增事件 `tool.execution` 记录补齐范围后的实际参数；`retrieval.batch`、`retrieval.progress`、`retrieval.finished` 记录新增证据、复盘、重复、轮次、查询、缓存、媒体及停止原因；`model.failed` 和 `run.failed` 保存模型调用阶段及经过脱敏、截断的诊断摘要，Provider 原始状态不会透传给用户；`tool.pairing_repaired` 记录出站前修补的调用 ID。`forum_http_requests` 单独统计论坛 GET，包括补取和重试。排查时先确认 `runtime.profile_used`，再分析调用轨迹。

沿用 `backward-compatible / recent:2` 发布策略，没有新增表。发布前仍须运行 Release 的旧镜像兼容性检查；本地 JSON 配置回归不能替代旧版本容器测试。回滚后旧程序读取归档的完整提示词，新模式的人设与补充要求不会自动转换成旧格式。发布、部署和线上发帖验收单独执行。
