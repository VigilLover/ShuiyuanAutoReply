# 上游合并报告 — 2025-05-02

## 目录

- [一、合并概览](#一合并概览)
- [二、目录结构变更](#二目录结构变更)
- [三、新增文件](#三新增文件)
- [四、逐文件函数级变更明细](#四逐文件函数级变更明细)
- [五、图片生成：两种实现对比](#五图片生成两种实现对比)
- [六、聊天模型：三种实现对比](#六聊天模型三种实现对比)
- [七、切换模型需要的代码修改](#七切换模型需要的代码修改)

---

## 一、合并概览

| 项目 | 内容 |
|------|------|
| 分支 | `merge-upstream-20250502`（基于 `dev`） |
| 上游 | `upstream/main`，13 个提交 |
| 合并方式 | `reset --soft` + 单次提交（`cc77229`） |
| 冲突次数 | 5 次（mention_chat_model × 4, mention_model × 1, .gitignore × 1, .env.example × 1） |
| 最终策略 | 保留上游所有架构改进，同时保留我们独有的模型接口和功能 |

---

## 二、目录结构变更

```
Before (dev)                          After (merge-upstream-20250502)
─────────────────────────────────     ─────────────────────────────────
example/                          →   examples/
example/main.py                   →   examples/main.py
example/backend.py                →   examples/backend.py（上游新增）
example/models/mention_chat_model.py → examples/models/mention_model/mention_chat_model.py
example/models/mention_model.py   →   examples/models/mention_model/mention_model.py
example/models/mention_tongyi_model.py → examples/models/mention_model/mention_tongyi_model.py
example/models/mention_pet_model.py →  examples/models/mention_model/mention_pet_model.py
example/models/mention_google_model.py → ╳ 上游删除（被 openrouter 替代）
example/models/image_generation.py →  examples/models/mention_model/image_generation.py
example/models/record_topic_model.py → examples/models/record_model/record_topic_model.py
example/models/stock_topic_model.py →  examples/models/stock_model/stock_topic_model.py
example/models/tarot_topic_model.py →  examples/models/tarot_model/tarot_topic_model.py
example/models/tarot_tongyi_model.py → examples/models/tarot_model/tarot_openrouter_model.py
                                    → examples/models/mention_model/shuiyuan_tools_wrapper.py（上游新增）
                                    → examples/models/mention_model/shuiyuan_tools_objects.py（上游新增）
                                    → examples/models/mention_model/mention_openrouter_model.py（上游新增）
assets/                           →  src/shuiyuan_auto_reply/assets/
tools/                            →  scripts/
                                    → src/shuiyuan_auto_reply/openrouter/（上游新增包）
src/shuiyuan_auto_reply/tongyi/   → ╳ 上游删除（不再需要）
.env_example                      → .env.example（统一命名）
```

---

## 三、新增文件

### 上游新增（从 upstream/main 引入）

| 文件 | 作用 |
|------|------|
| `src/shuiyuan_auto_reply/openrouter/__init__.py` | 包初始化 |
| `src/shuiyuan_auto_reply/openrouter/openrouter_model.py` | OpenRouter 客户端基础类 + 常量 + 环境变量解析 |
| `src/shuiyuan_auto_reply/openrouter/image_tool.py` | 基于 OpenRouter 的图片生成与上传 |
| `examples/models/mention_model/mention_openrouter_model.py` | OpenRouter 聊天模型实现 |
| `examples/models/mention_model/shuiyuan_tools_wrapper.py` | 水源 API 工具包装器（给 LLM agent 用） |
| `examples/models/mention_model/shuiyuan_tools_objects.py` | 精简数据类（PostShort、UserShort） |
| `examples/backend.py` | FastAPI 后端服务 |

### 我们保留的自有文件

| 文件 | 作用 |
|------|------|
| `examples/models/mention_model/mention_tongyi_model.py` | 通义千问聊天模型 |
| `examples/models/mention_model/mention_pet_model.py` | 宠物系统（rua 互动、情绪、结局） |
| `examples/models/mention_model/image_generation.py` | 自定义 API 图片生成工具 |

---

## 四、逐文件函数级变更明细

### 4.1 `mention_chat_model.py`（基础聊天模型）

| 方法 | 变更类型 | 说明 |
|------|----------|------|
| `__init__` | **保留自研** | 上游增加了 `DEFAULT_OPENROUTER_MAX_RETRIES` 导入，我们的 persona 系统（prompts_config、username 参数）保持不变 |
| `_load_mcp_tools` | **合并** | 上游新增 `"sse_read_timeout": 900`，我们的 SSE transport 保持不变 |
| `_load_shuiyuan_tools` | **合并** | 函数列表合并双方：上游的 `query_recent_posts_by_topic_id` + `generate_image_and_upload`，我们的 `search_posts_by_time_range_and_topic` + `generate_image`（本地图片工具）。上游改用了 `ShuiyuanToolsWrapper` 来代理函数调用 |
| `initialize_agent` | **合并** | 上游新增 `.with_retry(stop_after_attempt=DEFAULT_OPENROUTER_MAX_RETRIES)`，我们的 `return_intermediate_steps=True` 和日志保留 |
| `_get_recent_posts` | **上游删除** | 被 `get_recent_msgs_context` 中直接使用 `ShuiyuanToolsWrapper` 替代，我们采纳上游变更 |
| `get_recent_msgs_context` | **上游重写** | 改用 `ShuiyuanToolsWrapper.query_recent_posts_by_topic_id`，返回 `PostShort` 而非 `PostDetails`，内容截断到 192 字符 |
| `get_pumpkin_response` | **保留自研** | 我们的 `generate_image` 工具输出自动插入回复最前面的逻辑保留 |
| `parse_model_output` | 抽象方法 | 无变化 |

### 4.2 `mention_model.py`（行为调度器）

| 方法 | 变更类型 | 说明 |
|------|----------|------|
| `__init__` | **合并** | 上游初始化 `MentionOpenRouterModel`，我们保留 `MentionTongyiModel` + `MentionPetModel` + persona 配置。两者共存 |
| `_remove_shuiyuan_signature` | **上游删除** | 移至 `ShuiyuanModel.remove_shuiyuan_signature()` 静态方法，调用处改为 `ShuiyuanModel.remove_shuiyuan_signature()` |
| `_parse_prompt_text` | **合并** | 签名移除改为调用 `ShuiyuanModel.remove_shuiyuan_signature()` |
| `_pumpkin_condition` | **保留自研** | 使用 `mention_tongyi_model.get_pumpkin_response()` + 完整错误处理 + 签名附加。上游使用 `mention_openrouter_model`，我们的 Tongyi 优先 |
| `_clear_condition` | **合并** | 同时清除 `mention_tongyi_model` 和 `mention_openrouter_model` 的会话历史 |
| `_random_condition` | 不变 | — |
| `_poll_condition` | 不变 | — |
| `_help_condition` | 不变 | — |
| `_rua_condition` | 不变 | 使用 `pet_model.get_rua_response()` |
| `_new_action_routine` | **合并** | 上游将 `asyncio.gather()` 改为 `asyncio.create_task()` + `_bg_tasks` set 跟踪，我们的条件检查顺序保持不变 |

### 4.3 `shuiyuan_tools_wrapper.py`（上游新增，我们保留）

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `search_user_by_term(term)` | `List[UserShort] \| str` | 搜索用户，返回精简对象或错误信息 |
| `search_post_details_by_optional_username_topic(term, latest, username, topic_id)` | `List[PostShort] \| str` | 搜索帖子详情，按 topic 分组后展平返回 |
| `query_recent_posts_by_topic_id(topic_id, limit=10)` | `List[PostShort] \| str` | 获取指定 topic 的最近帖子 |
| `generate_image_and_upload(prompt)` | `str` | 通过 OpenRouter 生成图片并上传，返回短 URL |

所有方法均有 try/except 错误处理，失败时返回错误字符串而不是抛出异常。

### 4.4 `shuiyuan_tools_objects.py`（上游新增）

| 类 | 属性 | 说明 |
|----|------|------|
| `UserShort` | `username`, `name` | 从完整 User 对象精简 |
| `PostShort` | `id`, `post_number`, `topic_id`, `name`, `username`, `cooked[:192]`, `raw[:192]`, `reply_to_post_number`, `title` | 从完整 PostDetails 精简，`__str__` 中自动移除水源签名 |

### 4.5 `shuiyuan_model.py`（水源 API 模型）

| 方法 | 变更类型 | 说明 |
|------|----------|------|
| `remove_shuiyuan_signature` | **上游新增静态方法** | 使用 `re.sub(r'<div data-signature>.*?</div>', '', text, flags=re.DOTALL)` 统一移除签名 |
| `search_post_details_by_optional_username_topic` | **上游重构** | 返回值从 `List[PostDetails]` 变为 `Dict[str, List[PostDetails]]`（按 topic 标题分组）。API 端点从 `/search/query.json` 变为 `/search.json`。参数从 `params` 改为 `q` 查询字符串 |
| `query_recent_posts_by_topic_id` | **上游重构** | 返回值从 `List[PostDetails]` 变为 `Tuple[str, List[PostDetails]]`（增加 topic 标题） |
| `_TopicSearchResult` | **上游新增数据类** | 用于解析搜索响应中的 topic 信息 |

### 4.6 `topic_model.py` / `user_action_model.py`（后台任务）

| 变更 | 说明 |
|------|------|
| `asyncio.gather(*routines)` → `asyncio.create_task()` | 新帖处理不再阻塞 watcher 循环 |
| 新增 `_bg_tasks: set()` | 跟踪运行中的后台任务，防止被 GC |
| `task.add_done_callback(lambda t, s=self._bg_tasks: s.discard(t))` | 任务完成后自动清理 |

### 4.7 塔罗和运势

| 文件 | 变更 |
|------|------|
| `tarot_group_data.py` | 新增 5 个牌阵类：`SingleCardGuidanceGroup`(1张)、`ChoiceTarotGroup`(5张)、`FiveElementsGroup`(5张)、`HorseshoeGroup`(7张)、`CelticCrossGroup`(10张) |
| `fortune/constants.py` | 新增 15+ 条运势 to_do 条目，优化现有条目文本 |
| `tarot_group_model.py` | 去除 debug `print()`，使用集中化 `max_retries` |

### 4.8 `openrouter/openrouter_model.py`（上游新增）

```python
DEFAULT_OPENROUTER_MAX_RETRIES = 5
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-flash-lite-preview"

def openrouter_headers() -> dict[str, str]   # 返回 OpenRouter 所需的 HTTP 头
def openrouter_model(env_name, default) -> str  # 按优先级解析模型名称

class BaseOpenRouterModel:
    def __init__(self, *, api_key, base_url, max_retries)
    # 初始化 AsyncOpenAI 客户端，配置 OpenRouter 端点
```

---

## 五、图片生成：两种实现对比

当前代码中**同时存在两个图片生成实现**，在 `mention_chat_model._load_shuiyuan_tools` 中分别注册为两个工具：

| | 自研 `image_generation.py` | 上游 `openrouter/image_tool.py` |
|------|---------------------------|-----------------------------------|
| **注册名** | `generate_image` | `generate_image_and_upload` |
| **API** | 自定义 OpenAI 兼容端点 | OpenRouter API |
| **模型** | `IMAGE_GEN_MODEL`（默认 `gpt-image-2-pro`） | `OPENROUTER_IMAGE_MODEL`（默认 `google/gemini-3.1-flash-image-preview`） |
| **响应格式** | Markdown 文本中的 `![...](url)` | 结构化响应中的 base64 data URL |
| **下载** | 需要单独 HTTP GET 下载 | 无需下载，直接解码 base64 |
| **图片处理** | 不处理 | PIL 转 JPEG Q95 |
| **本地备份** | 总是保存到 `assets/generated_images/` | 仅当指定 `output_dir` 时保存 |
| **上传方法** | `try_upload_image`（带 base64 回退重试） | `upload_image`（直接上传） |
| **返回值** | `![image](url)` Markdown 链接 | 纯短 URL 字符串 |
| **宽高比/尺寸** | 不支持 | 支持 aspect_ratio、image_size |
| **结构** | 工厂函数 | 完整类（继承 `BaseOpenRouterModel`） |

### 如果要使用自定义图片 API 替换上游实现

在 `shuiyuan_tools_wrapper.py` 的 `generate_image_and_upload` 方法中，当前调用的是 `OpenRouterImageTool`。要改为使用我们自己的 `image_generation` 实现，需要修改该方法的内部实现。详见[第七节](#七切换模型需要的代码修改)。

---

## 六、聊天模型：三种实现对比

| | `MentionTongyiModel`（自研） | `MentionOpenRouterModel`（上游） |
|------|------------------------------|--------------------------------------|
| **继承** | `MentionChatModel` | `MentionChatModel` |
| **LLM 类** | `ChatTongyi` (langchain_community) | `ChatOpenAI` (langchain_openai) |
| **模型** | `qwen3-max` | `google/gemini-3.1-flash-lite-preview` |
| **API 密钥** | `DASHSCOPE_API_KEY` | `OPENROUTER_API_KEY` |
| **温度** | `1.5` | `0.8` |
| **思考模式** | `enable_thinking=True` | 无 |
| **重试** | Agent 级别 `.with_retry(stop_after_attempt=5)` | 类级别 `max_retries=5` + Agent 级别重试 |
| **parse_model_output** | 简单 `raw_output.strip()` | 处理 None/list/dict/对象等多种格式 |

**当前使用哪个？**

`mention_model.py` 的 `_pumpkin_condition` 中**使用 `mention_tongyi_model`**（Tongyi/Qwen），`mention_openrouter_model` 已初始化但未在分派链中使用。`_rua_condition` 使用独立的 `mention_pet_model`。

---

## 七、切换到不同模型需要的代码修改

### 场景 A：从 Tongyi 切换到 OpenRouter 聊天模型

修改 `examples/models/mention_model/mention_model.py` 的 `_pumpkin_condition` 方法：

```python
# 当前代码（第 ~108 行）使用 Tongyi：
reply = await self.mention_tongyi_model.get_pumpkin_response(topic_id, raw, user)

# 改为使用 OpenRouter：
reply = await self.mention_openrouter_model.get_pumpkin_response(topic_id, raw, user)
```

同时需要在 `.env` 中配置：
```
OPENROUTER_API_KEY=sk-or-xxx
OPENROUTER_MENTION_MODEL=google/gemini-3.1-flash-lite-preview
```

### 场景 B：从自定义图片 API 切换到 OpenRouter 图片生成

在 `examples/models/mention_model/shuiyuan_tools_wrapper.py` 的 `generate_image_and_upload` 方法中：

```python
# 当前代码使用上游 OpenRouterImageTool：
async def generate_image_and_upload(self, prompt: str) -> str:
    try:
        return await self.image_tool.generate_and_upload(
            prompt, output_dir="generated_images", image_size="1K",
        )
    except Exception as e:
        return str(e)
```

上游实现已在上方，如果需要用我们自己的 API，替换为：

```python
async def generate_image_and_upload(self, prompt: str) -> str:
    try:
        image_bytes = await self._generate_via_custom_api(prompt)
        response = await self.shuiyuan_model.try_upload_image(
            image_bytes, try_base64=True, try_base64_size_kb=40
        )
        return response.data
    except Exception as e:
        return f"图片生成失败: {e}"
```

### 场景 C：使用我们自己的图片 API（image_generation.py）

当前 `image_generation.py` 作为独立的 `generate_image` 工具已注册在 agent 中。LLM 可以选择调用 `generate_image`（我们的实现）或 `generate_image_and_upload`（上游实现）。

如果需要**只用我们的实现**，删除 `_load_shuiyuan_tools` 函数列表中的 `"generate_image_and_upload"`：

```python
# 在 mention_chat_model.py 的 _load_shuiyuan_tools 中：
function_list = [
    "search_user_by_term",
    "search_post_details_by_optional_username_topic",
    "query_recent_posts_by_topic_id",
    "search_posts_by_time_range_and_topic",
    # "generate_image_and_upload",  # 注释掉上游的
]
```

### 场景 D：让 MentionModel 支持多 LLM 切换

在 `mention_model.py` 的 `__init__` 中增加一个配置项：

```python
def __init__(self, model, bot_username, persona, llm_backend="tongyi"):
    ...
    if llm_backend == "openrouter":
        self.active_llm = self.mention_openrouter_model
    else:
        self.active_llm = self.mention_tongyi_model
```

然后在 `_pumpkin_condition` 中使用 `self.active_llm` 代替硬编码的 `self.mention_tongyi_model`。

### 关键环境变量汇总

```bash
# ===== OpenRouter（上游） =====
OPENROUTER_API_KEY=           # OpenRouter API 密钥
OPENROUTER_MODEL=google/gemini-3.1-flash-lite-preview  # 全局默认模型
OPENROUTER_MENTION_MODEL=     # 聊天模型（覆盖全局默认）
OPENROUTER_TAROT_MODEL=       # 塔罗解读模型
OPENROUTER_TAROT_GROUP_MODEL= # 塔罗牌阵选择模型
OPENROUTER_IMAGE_MODEL=google/gemini-3.1-flash-image-preview  # 图片生成模型

# ===== 通义千问 / DashScope（自研） =====
DASHSCOPE_API_KEY=            # 阿里云 DashScope API 密钥

# ===== 自定义图片生成（自研） =====
IMAGE_GEN_API_KEY=            # 图片 API 密钥
IMAGE_GEN_API_URL=https://www.openclaudecode.cn/v1/chat/completions  # 图片 API 端点
IMAGE_GEN_MODEL=gpt-image-2-pro  # 图片模型

# ===== 基础设施 =====
MCP_SERVER_URL=http://localhost:8000/sse  # MCP 服务器地址
NEO4J_DB_URL=bolt://localhost:7687        # Neo4j 数据库
NEO4J_DB_AUTH=("neo4j", "password")       # Neo4j 认证
HF_ENDPOINT=https://hf-mirror.com         # HuggingFace 镜像
```
