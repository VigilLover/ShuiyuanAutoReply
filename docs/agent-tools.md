# Agent 资料与参考素材

论坛与网页共用通用工具；网页的论坛写权限保持关闭。此次更新不新增数据库表，本轮证据、查询缓存、素材集在请求结束后释放，不是长期记忆。

## 帖子和用户

- `get_post(topic_id, post_number, refresh=False, cursor=0)` 根据话题内楼层读取；`get_post_by_id(post_id, refresh=False, cursor=0)` 使用全站帖子 ID。不能互换两种编号。
- 精准读取优先获取 `raw`，缺失时补取详情，再回退为清理后的 HTML 文本。结果包含 `content_source`、回复关系、正文／引用提及、图片、来源和警告。
- 精准正文每页 12000 字符，`truncated`、`next_cursor` 明确指示后续内容。用同一精准工具的 `cursor`，或 `read_tool_result(result_id, cursor)` 读取后续页。
- 搜索正文摘要最多 800 字符，并说明返回结果不保证穷尽。标题可为空；不为标题增加整话题请求。
- `get_user` 精确查用户名，`get_users` 批量查 1–50 个用户名，并发最多 4。输出保留输入映射和逐项错误。只在需要头像时传 `include_avatar=True`。
- `search_user` 保留模糊发现用途。旧 `search_user_by_id` 依赖用户发帖记录，查不到不代表不存在。
- 相同成功查询仅在当前轮复用；`refresh=True` 请求新数据。批量重试不会重复请求成功用户。非重试性 HTTP 错误不会盲目重试。

## 上下文和工具配置

`common.runtime.context_token_budget` 默认 24000，必须为正值；它是动态文本输入的近似 token 预算，不是模型的真实总窗口大小。工具 schema、静态规则和提供商多模态配额独立处理。当前指令、调用／返回配对等必要信息可能使极端输入超过估算预算。

压缩只修改发送给模型的视图：完整结果在本轮内可回读，目标帖引用、最近两组调用和实体映射仍可访问。不设置按任务类型的搜索次数限制。

默认工具目录包含新增工具。已有显式 `enabled_tools` 白名单不会自动扩大；需要使用新能力时，在管理界面的论坛／网页工具设置中启用 `get_user`、`get_users`、`get_post_by_id`、`read_tool_result`、`prepare_image_references`。若未启用 `read_tool_result`，运行时保留完整工具消息，避免产生不可回读的压缩内容；精准读帖仍可用 `cursor` 翻页。

## 参考图片

`prepare_image_references(references)` 接受 1–50 项，每项包含唯一 `key`、`url` 和可选 `label`。返回本轮 `reference_set_id`、成功顺序、逐项错误。成功下载会复用；超时、连接失败、429、5xx 最多尝试 3 次，遵循 Retry-After 和本轮剩余时间。已确认用户的头像 404 时可刷新一次资料，只有 URL 改变后才重试。

`generate_image(..., reference_set_id=...)` 用成功素材建立实际编号，prompt 使用标签描述对象。部分失败时只纳入成功素材对应对象，并在最终回复说明未纳入项；全部失败不会改成无参考生成。旧 `reference_images` URL 列表继续支持，但部分失败会先返回准备结果，要求模型调整描述后再生成，防止旧编号错位。

素材集仅在当前轮有效。新一轮修改旧图时需要重新准备。图像生成成功不表示内容已通过身份、人数或细节核验。

## 验证与观察

默认离线测试包含模拟 HTTP 服务、中文 URL 编码、签名查询串保留、部分失败、编号与图片字节顺序、超长正文、216 条历史压缩、并发轮隔离以及真实 LangGraph 编排测试。运行 `uv run pytest -q`；仅已安装环境验证可使用 `uv run --no-sync pytest -q`。

观察 `context.evidence` 的结果数及缓存命中数、`image.references_prepared` 的成功／失败计数，以及上下文投影的估算 token 日志。线上验证应另外安排，不在离线测试中调用论坛或付费生图服务。
