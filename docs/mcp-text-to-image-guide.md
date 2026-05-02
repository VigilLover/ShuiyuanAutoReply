# MCP 文生图工具开发指南

> 参考上游实现：[https://github.com/Hydroiodic/ShuiyuanAutoReply](https://github.com/Hydroiodic/ShuiyuanAutoReply)
> 本地路径：`/Users/qianminhao/SoftwareTools/ShuiyuanAutoReply-upstream` (main 分支 + mcp 分支)

## 1. 上游仓库结构概览

上游将 MCP Server 和 Bot Client 放在同一仓库的不同分支：

| 分支 | 角色 | 内容 |
|------|------|------|
| `mcp` | MCP Server | 提供工具（文生图、Docker 代码执行等） |
| `main` | Bot Client | 论坛机器人，通过 langchain-mcp-adapters 连接 MCP Server |

### 1.1 mcp 分支目录结构

```
src/
├── __init__.py
├── instance.py           # FastMCP 全局单例
└── tools/
    ├── __init__.py        # 通过环境变量控制工具加载
    ├── image_tool.py      # 文生图工具（DashScope Qwen Image）
    └── command_tool.py    # Docker 沙箱代码执行工具
example/
└── main.py                # 启动入口：mcp.run(transport="streamable-http")
```

### 1.2 main 分支目录结构

```
src/shuiyuan_auto_reply/   # Bot 核心库
example/
├── main.py                # Bot 启动入口
└── models/
    ├── mention_chat_model.py   # Agent 基类（MCP 工具加载 + LangChain Agent）
    ├── mention_google_model.py # Gemini 模型实现
    ├── mention_tongyi_model.py # 通义模型实现
    ├── mention_model.py        # Mention 业务逻辑（条件匹配 + 回复分发）
    └── tarot_topic_model.py    # 塔罗牌图片上传示例
```

---

## 2. 上游参考实现详解

### 2.1 MCP Server 实例化 (`src/instance.py`)

```python
from mcp.server.fastmcp import FastMCP

# 全局单例，json_response=True 表示返回值直接 JSON 序列化
mcp = FastMCP("ShuiyuanAutoReply", json_response=True)
```

### 2.2 文生图工具完整实现 (`src/tools/image_tool.py`)

**这是上游的核心参考实现，使用阿里云 DashScope 百炼平台的 Qwen Image 模型：**

```python
import os
from typing import Any

import dashscope
from dashscope import MultiModalConversation

from ..instance import mcp

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY is not configured.")


def _extract_image_base64(response: Any) -> str:
    """从 DashScope 响应中提取图片 base64 数据（去除 data:image/xxx;base64, 前缀）"""
    if getattr(response, "status_code", 200) != 200:
        message = getattr(response, "message", None) or getattr(response, "code", None)
        raise RuntimeError(f"DashScope request failed: {message or response}")

    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")

    choices = getattr(output, "choices", None)
    if choices is None and isinstance(output, dict):
        choices = output.get("choices")
    if not choices:
        raise RuntimeError("DashScope response did not include any image choices.")

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not content:
        raise RuntimeError("DashScope response did not include image content.")

    for item in content:
        image = getattr(item, "image", None)
        if image is None and isinstance(item, dict):
            image = item.get("image")
        if isinstance(image, str) and image:
            # 去除 "data:image/png;base64," 前缀，只保留纯 base64
            return image.split(",", 1)[1] if image.startswith("data:image/") else image

    raise RuntimeError("DashScope response content did not contain a base64 image.")


@mcp.tool()
async def generate_image(text: str) -> str:
    """
    Generate an image from a text prompt with DashScope Qwen Image.
    NOTE: The user cannot see how you generate the image, so you must add "![](image-url)"
        (Markdown image format) in your final response.

    Args:
        text: The image prompt.

    Returns:
        The generated image URL.
    """

    messages = [
        {
            "role": "user",
            "content": [{"text": text}],
        }
    ]

    try:
        response = MultiModalConversation.call(
            api_key=api_key,
            model="qwen-image-2.0",
            messages=messages,
            result_format="message",
            stream=False,
            n=1,
            watermark=True,
            negative_prompt="",
        )
        return _extract_image_base64(response)
    except Exception as exc:
        return f"Error: {exc}"
```

### 2.3 工具加载控制 (`src/tools/__init__.py`)

通过环境变量按需启用工具，避免未配置时启动失败：

```python
import logging
import os
import dotenv

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

def _tool_enabled(name: str) -> bool:
    return os.getenv(name, "False").lower() in {"1", "true", "yes", "on"}

if _tool_enabled("IMAGE_TOOL_ENABLED"):
    try:
        from . import image_tool
    except Exception as exc:
        logger.warning("Image tool disabled: %s", exc)

if _tool_enabled("COMMAND_TOOL_ENABLED"):
    from . import command_tool
```

### 2.4 MCP Server 启动入口 (`example/main.py`)

```python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dotenv
from src.instance import mcp

dotenv.load_dotenv()

# 导入 tools 包以触发工具注册（副作用导入）
from src import tools

if __name__ == "__main__":
    # 使用 streamable-http 传输（非 SSE）
    mcp.run(transport="streamable-http")
```

### 2.5 Bot 端连接 MCP Server (`mention_chat_model.py` 上游版本)

上游 main 分支的 Bot 使用 **`http`** 传输（匹配 MCP Server 的 `streamable-http`）：

```python
async def _load_mcp_tools(self, url: str) -> List[StructuredTool]:
    client = MultiServerMCPClient({
        "default": {
            "transport": "http",   # ← 与当前项目不同（当前用 "sse"）
            "url": url,
        }
    })
    mcp_tools = await client.get_tools()
    return mcp_tools
```

当前项目（dev 分支）使用的是 `"sse"` 传输，如果你需要连接上游风格的 MCP Server，需要改为 `"http"` 或 `"streamable-http"`。

---

## 3. 返回值格式分析

### 3.1 上游方案：返回纯 base64 字符串

上游 `generate_image()` 返回 `str`（纯 base64），FastMCP 会将其转为 `TextContent`：

```
MCP Tool 返回 "iVBORw0KGgo..." (base64 string)
  → _convert_to_content() 输出 TextContent(type="text", text="iVBORw0KGgo...")
    → langchain-mcp-adapters 转为 {"type": "text", "text": "iVBORw0KGgo..."}
      → LLM 收到一段很长的 base64 字符串
```

**问题**：LLM（非多模态模型）看到的是无意义的 base64 字符，需要 Bot 端做额外处理。

### 3.2 改进方案：返回 ImageContent

使用 MCP SDK 的 `Image` 辅助类，让图片以 `ImageContent` 传输：

```python
from mcp.server.fastmcp.utilities.types import Image

@mcp.tool()
async def generate_image(text: str) -> Image:
    response = MultiModalConversation.call(...)
    b64_data = _extract_image_base64(response)
    image_bytes = base64.b64decode(b64_data)
    return Image(data=image_bytes, format="png")
```

这样经过 FastMCP 的 `_convert_to_content()` 会产生 `ImageContent`，langchain-mcp-adapters 会转为 `ImageContentBlock`（`{"type": "image", "base64": "...", "mime_type": "image/png"}`）。

**多模态模型（如 Gemini）能直接"看到" ImageContentBlock。**

### 3.3 返回值格式对照

| 工具返回 | FastMCP 转换后 (ContentBlock) | langchain-mcp-adapters 转换后 | 适用场景 |
|---------|------------------------------|------------------------------|---------|
| `str` | `TextContent(type="text", text=...)` | `{"type": "text", "text": "..."}` | 文本结果，LLM 直接阅读 |
| `Image(data=bytes)` | `ImageContent(type="image", data=<b64>, mimeType=...)` | `{"type": "image", "base64": "...", "mime_type": "..."}` | 多模态 LLM 直接看 |
| `CallToolResult(content=[TextContent, ImageContent])` | 保持原样 | `[{text...}, {image...}]` | 同时返回文字和图片 |
| `list[Image]` | `[ImageContent, ImageContent, ...]` | `[{image...}, {image...}]` | 多张图片 |
| `BaseModel` (Pydantic) | `TextContent(type="text", text=<json>)` | `{"type": "text", "text": "{...}"}` | 结构化数据 |

---

## 4. Bot 端改造方案

### 4.1 当前 Bot 的局限

`MentionChatModel.get_pumpkin_response()` 返回 `Optional[str]`，`parse_model_output()` 只提取文本。即使 MCP 工具返回了图片，图片数据也会在 `parse_model_output()` 阶段被丢弃。

### 4.2 推荐方案：从 intermediate_steps 提取图片并上传

不改变 LLM 推理流程，在最终回复中追加图片。参考上游 `tarot_topic_model.py` 的图片上传模式（已有 `try_upload_image()`）：

```python
import base64
from langchain.agents import AgentExecutor

# 在 AgentExecutor 中设置 return_intermediate_steps=True
self.agent_executor = AgentExecutor(
    agent=agent,
    tools=all_tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,  # ← 新增
)
```

修改 `get_pumpkin_response()`：

```python
async def get_pumpkin_response(
    self, topic_id: int, conversation: str, user: User
) -> Optional[str]:
    if not self.agent_executor:
        await self.initialize_agent()

    # ... 现有逻辑 ...

    response = await self.agent_executor.ainvoke(agent_input)

    raw_output = response.get("output")
    final_clean_text = self.parse_model_output(raw_output)

    # === 新增：从 intermediate_steps 中提取生成/返回的图片 ===
    intermediate_steps = response.get("intermediate_steps", [])
    image_markdowns = []
    for action, tool_output in intermediate_steps:
        if action.tool == "generate_image":
            image_markdown = await self._process_tool_output_for_image(tool_output)
            if image_markdown:
                image_markdowns.append(image_markdown)

    if image_markdowns:
        final_clean_text += "\n\n" + "\n".join(image_markdowns)

    # ... 后续逻辑 ...
```

新增辅助方法处理两种返回值格式：

```python
async def _process_tool_output_for_image(self, tool_output) -> Optional[str]:
    """
    处理工具返回的图片数据，上传到 Shuiyuan，返回 Markdown。

    兼容两种格式：
    - ImageContentBlock: {"type": "image", "base64": "...", "mime_type": "..."}
    - TextContentBlock:  {"type": "text", "text": "<base64 string>"}
    """
    if isinstance(tool_output, list):
        for block in tool_output:
            result = await self._process_single_block(block)
            if result:
                return result
    return None

async def _process_single_block(self, block) -> Optional[str]:
    """处理单个 content block，识别图片数据并上传"""
    if not isinstance(block, dict):
        return None

    if block.get("type") == "image":
        # ImageContentBlock: 直接的图片 base64
        image_b64 = block.get("base64")
        image_bytes = base64.b64decode(image_b64)
        image_url = await self.model.try_upload_image(image_bytes)
        return image_url.data  # "![img](short_url)" 或 base64 HTML

    if block.get("type") == "text":
        # TextContentBlock: 可能是纯 base64 字符串（上游方案）
        # 尝试检测是否为 base64 图片数据
        text = block.get("text", "")
        if len(text) > 100 and not text.startswith("Error"):
            try:
                image_bytes = base64.b64decode(text)
                # 验证是否为有效图片（检查 magic bytes）
                if image_bytes[:4] in (b'\x89PNG', b'\xff\xd8\xff', b'RIFF'):
                    image_url = await self.model.try_upload_image(image_bytes)
                    return image_url.data
            except Exception:
                pass  # 不是 base64 图片，跳过

    return None
```

### 4.3 关于 System Prompt

LLM 需要知道它可以调用 `generate_image` 工具，并且需要在回复中用 Markdown 格式引用图片。参考上游的描述：

```
NOTE: The user cannot see how you generate the image, so you must add
"![](image-url)" (Markdown image format) in your final response.
```

如果在 Bot 端自动追加图片，可以修改为：

```
你可以使用 generate_image 工具来根据文字描述生成图片。
当用户要求画图或生成图片时，使用此工具。
生成图片后，请用文字简要描述生成的图片内容。图片会自动附加到回复末尾。
```

---

## 5. 环境配置

### 5.1 MCP Server 端 (`.env`)

```bash
# 阿里云 DashScope API Key
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx

# 工具开关
IMAGE_TOOL_ENABLED=true
COMMAND_TOOL_ENABLED=false
```

### 5.2 Bot 端 (`.env`)

```bash
# MCP Server 地址（注意传输协议）
# 如果 MCP Server 使用 streamable-http：
MCP_SERVER_URL=http://localhost:8000/mcp
# 如果 MCP Server 使用 SSE：
# MCP_SERVER_URL=http://localhost:8000/sse
```

### 5.3 Transport 协议对应关系

| MCP Server 启动方式 | Bot 端 transport 参数 |
|---------------------|----------------------|
| `mcp.run(transport="streamable-http")` | `"http"` |
| `mcp.run(transport="sse")` | `"sse"` |
| `mcp.run()` (默认) | `"sse"` |

---

## 6. 开发步骤

### 第一步：确认 transport 协议一致

1. 在 MCP Server 项目中确定使用 `streamable-http` 还是 `sse`
2. 在 Bot 的 `_load_mcp_tools()` 中设置对应的 transport 参数
3. 用 `test/test_mcp.py` 验证连接

### 第二步：实现 MCP Server 端文生图工具

1. 参照 2.2 节的参考实现（或改用 `Image` 返回值）
2. 通过环境变量控制工具启用
3. 本地启动并验证

### 第三步：改造 Bot 端

1. 在 `initialize_agent()` 中设置 `return_intermediate_steps=True`
2. 在 `get_pumpkin_response()` 中添加图片提取和上传逻辑
3. 利用已有的 `try_upload_image()` 上传图片到 Shuiyuan

### 第四步：调整 System Prompt

告诉 LLM 工具的存在和图片的处理方式。

---

## 7. 关键源文件索引

### 上游参考实现

| 文件 | 作用 |
|------|------|
| `upstream/src/instance.py` | FastMCP 全局单例 (`json_response=True`) |
| `upstream/src/tools/image_tool.py` | 文生图工具实现（DashScope Qwen Image） |
| `upstream/src/tools/__init__.py` | 环境变量控制工具加载 |
| `upstream/src/tools/command_tool.py` | Docker 沙箱代码执行工具（参考工具定义模式） |
| `upstream/example/main.py` | MCP Server 启动入口 (`streamable-http`) |
| `upstream/example/models/mention_chat_model.py` | Bot Agent 基类（MCP 工具加载 + LangChain 集成） |
| `upstream/example/models/mention_google_model.py` | Gemini 模型实现 + parse_model_output 示例 |

### 当前项目

| 文件 | 作用 |
|------|------|
| `example/models/mention_chat_model.py:141-228` | Bot 端 MCP 工具加载流程（当前用 SSE） |
| `example/models/mention_chat_model.py:287-333` | `get_pumpkin_response()` — Bot 回复主流程 |
| `example/models/mention_google_model.py:41-74` | parse_model_output 示例 |
| `src/shuiyuan_auto_reply/shuiyuan/shuiyuan_model.py:360-439` | `upload_image()` / `try_upload_image()` — 图片上传 |
| `example/models/tarot_topic_model.py:45-80` | 塔罗牌图片上传示例（参考模式） |
| `test/test_mcp.py` | MCP 连接测试 |

### SDK 内部实现

| 文件 | 作用 |
|------|------|
| `mcp/server/fastmcp/utilities/types.py:9-54` | `Image` 辅助类 |
| `mcp/server/fastmcp/utilities/func_metadata.py:499-533` | `_convert_to_content()` — 返回值 → ContentBlock |
| `mcp/types.py:1041-1058` | `ImageContent` 协议类型 |
| `langchain_mcp_adapters/tools.py:84-198` | MCP ContentBlock → LangChain Block 转换 |
