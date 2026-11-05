# 测试冗余审查

审查基线：`274fe0f`，日期：2026-11-05。

审查范围为 `test/` 的 43 个测试文件及 `conftest.py`、`web/test/` 的 7 个测试文件，并核对 CI、脚本和文档中的引用。清理前 pytest 收集 350 项。静态比对未发现跨文件完全相同的测试体；以下删除依据来自收集结果、调用链和断言，而非文件名或长度。

## 删除与合并

| 文件或用例 | 处理与依据 |
| --- | --- |
| `test/test_openai_images_api.py` | 删除。文件没有任何 pytest 用例，也没有项目内调用者；实际是独立手工生图脚本，使用说明还指向不存在的 `scripts/test_openai_images_api.py`。其编辑请求自行构造 JSON，未走当前生产服务使用的 multipart 上传链路，不能作为生产接口回归。生产链路已有 `test_image_generation.py` 的模拟 HTTP 测试。 |
| `test/test_mcp.py` | 删除。缺少 URL 时直接返回，连接异常被捕获后只写日志，返回空工具列表也不失败；整个调用链没有断言。这是始终可能通过的连通性诊断，不构成自动化验收。MCP 工具解析、绑定和部署约束继续由专门套件覆盖。 |
| `test/test_mention_tool_trim.py` | 删除文件，将 3 个独立的超预算/证据投影回归及其 helper 原样移入 `test_context_budget.py`。三个“小结果保持不变”的兼容入口用例并入该文件已有的预算内用例，分别保留单轮、十轮、四轮并行四工具的输入场景，以 subtest 同时验证投影函数和兼容入口。 |
| `test_deepseek_vision.py::VisionContractTests::test_deepseek_payload_preserves_file_and_image_url_blocks` | 删除重复用例。`test_deepseek_protocol_contracts.py` 的请求合同用例已覆盖相同图片块，并额外检查思考字段回传、参数与 token 限制；将原用例的 `role == user` 断言一并保留在合同用例中。 |

清理不修改生产代码或前端构建资源。迁移的 3 个投影回归在移动前后做了 AST 等价比对。

## 保留套件与独立职责

下表列出其余全部测试文件。Python 文件均位于 `test/`，前端文件位于 `web/test/`。

| 职责 | 保留文件 | 保留理由 |
| --- | --- | --- |
| 架构、端口与生命周期 | `test_architecture_boundaries.py`、`test_architecture_core.py`、`test_port_contracts.py`、`test_lifecycle.py`、`test_scheduling.py` | 分别保护依赖方向、业务分派/会话身份、适配器转发、资源释放和并发调度；服务单测不能替代 HTTP 或真实任务生命周期回归。 |
| 配置与提示词 | `test_deployment_config.py`、`test_settings_contract.py`、`test_model_configs.py`、`test_managed_profiles.py`、`test_prompt_repository.py` | 分别覆盖部署配置优先级、环境变量文档、运行时切换/事务、提示词迁移和打包文本合同。快照与迁移测试的断言目标不同。 |
| API、持久化与论坛任务 | `test_api_integration.py`、`test_stage2_web_state.py`、`test_forum_monitor.py` | 兼容接口的认证/状态码、SQLite 历史/事件及管理 API、论坛发布状态机均有独立失败场景。 |
| 论坛与外部工具 | `test_forum_auth.py`、`test_unified_tools.py`、`test_mcp_tool_adapters.py` | 登录 Cookie/CSRF、论坛工具分页游标、MCP 结果格式与证据分别属于不同边界。 |
| 模型与向量服务 | `test_ai_models.py`、`test_deepseek_protocol_contracts.py`、`test_deepseek_responses_live.py`、`test_deepseek_vision.py`、`test_pet_model.py`、`test_remote_embeddings.py` | 模型装配、协议映射、真实 Responses 接口、图片输入、宠物状态和向量排序/重试各有独立覆盖。Chat Completions 连通性与 Responses 连通性不能互相替代；live 测试默认跳过并不意味着冗余。 |
| 图片处理与交付 | `test_image_generation.py`、`test_image_references.py`、`test_media_limits.py`、`test_image_handles.py`、`test_forum_media.py`、`test_on_demand_media.py` | 分别覆盖生图传输、参考素材准备、解码限额、句柄生命周期、论坛上传和按需加载/终态失败共享。 |
| Agent 上下文与收敛 | `test_context_budget.py`、`test_task_progress.py`、`test_turn_results.py`、`test_tool_call_pairing.py`、`test_retrieval_control.py`、`test_convergence_regressions.py`、`test_forum_agent_flow.py`、`test_retrieval_failure_replay.py` | 预算投影、证据身份、请求缓存/超时、工具配对、收敛规则、节点回归、真实图编排和历史故障重放覆盖不同层次，不能因使用相同 helper 就整文件删除。 |
| 迁移、CI 与发布 | `test_data_migration.py`、`test_ci_integration.py`、`test_mcp_deployment.py`、`test_release_deployment.py` | 数据导出、容器健康探测重试、MCP 构建输入和发布/回滚安全分别约束不同运维路径。 |
| 前端 | `bbcode.test.ts`、`conversations.test.ts`、`forum.test.ts`、`model-configs.test.ts`、`prompt.test.ts`、`reasoning.test.ts`、`runs.test.ts` | 分别覆盖文本转换、失败后重试输入保留、论坛事件归并、配置表单、提示词状态、真实 Vue 折叠行为和轮次统计，没有可以整文件删除的重复覆盖。 |

公共 `conftest.py` 和被多个套件及评估脚本使用的 `OfflineChat` 保留。未将“断言数量少”“测试默认跳过”“与集成测试调用同一函数”单独作为删除依据。

## 验证

验证结果：

- pytest 收集项：350 → 345。对迁移用例的路径归一化后，只有预期的 5 个用例标识移除，没有额外丢失的用例。
- Python 测试文件：43 → 40，公共 `conftest.py` 保留。
- 默认离线 pytest：335 passed、10 skipped，另有 17 个 subtest 通过。包含需要本机临时 HTTP 服务的图片传输回归；未启用 live 测试。
- 前端：30 项测试通过，7 个测试文件全部保留。
- 全量 Black/isort 和 `git diff --check` 通过；删除文件没有遗留的导入或脚本引用。

本次未测量新的行覆盖率；保留独立回归的判断依据是输入场景、执行层次、具体断言、收集清单及迁移代码的 AST 等价比对。
