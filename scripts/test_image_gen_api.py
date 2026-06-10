#!/usr/bin/env python3
"""独立的图片生成 API 验证脚本。

使用 .env 中的凭证测试 Right.Codes 生图 API：
1. 发送生图请求
2. 解析响应提取图片 URL
3. 下载图片到本地

用法:
    cd /Users/qianminhao/SoftwareTools/ShuiyuanAutoReply
    python scripts/test_image_gen_api.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

# 加载 .env
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_image_gen")


async def test_images_endpoint():
    """测试 /v1/images/generations 端点"""
    api_key = os.getenv("IMAGE_GEN_API_KEY", "").strip()
    api_url = "https://www.right.codes/draw/v1/images/generations"
    model = os.getenv("IMAGE_GEN_MODEL", "gpt-image-2").strip()

    if not api_key:
        logger.error("IMAGE_GEN_API_KEY 未配置")
        return False

    payload = {
        "model": model,
        "prompt": "一只可爱的柴犬在草地上奔跑，阳光明媚，二次元插画风格",
        "size": "1024x1024",
        "response_format": "url",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info("=" * 60)
    logger.info("测试 1: /v1/images/generations 端点")
    logger.info("请求 payload: %s", json.dumps(payload, ensure_ascii=False))

    started = time.monotonic()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600, connect=30),
            ) as resp:
                elapsed = time.monotonic() - started
                body = await resp.text()

                logger.info("HTTP %d, 耗时 %.2fs, 响应大小 %d bytes", resp.status, elapsed, len(body))

                if resp.status != 200:
                    logger.error("❌ 请求失败: %s", body[:500])
                    return False

                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    logger.error("❌ 响应不是有效 JSON: %.500s", body)
                    return False

                logger.info("响应结构 keys: %s", list(data.keys()))
                logger.info("响应完整内容: %s", json.dumps(data, ensure_ascii=False, indent=2)[:1000])

                # 提取图片 URL
                image_url = None
                try:
                    image_url = data["data"][0]["url"]
                except (KeyError, IndexError, TypeError):
                    pass

                if not image_url:
                    logger.error("❌ 未在 data[0].url 中找到图片 URL")
                    return False

                logger.info("✅ 提取到图片 URL: %s", image_url)

    except Exception as exc:
        logger.error("❌ 请求异常: %s", exc)
        return False

    # 下载图片
    logger.info("正在下载图片...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    logger.error("❌ 下载图片失败: HTTP %d", resp.status)
                    return False
                image_bytes = await resp.read()
                logger.info("✅ 下载成功: %d bytes", len(image_bytes))

                # 保存到 assets/generated_images
                output_dir = project_root / "assets" / "generated_images"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"test_{timestamp}.png"
                output_path.write_bytes(image_bytes)
                logger.info("✅ 图片已保存到: %s (%.1f KB)", output_path, len(image_bytes) / 1024)
    except Exception as exc:
        logger.error("❌ 下载图片异常: %s", exc)
        return False

    return True


async def test_chat_completions_endpoint():
    """测试 /v1/chat/completions 端点（兼容模式）"""
    api_key = os.getenv("IMAGE_GEN_API_KEY", "").strip()
    api_url = "https://www.right.codes/draw/v1/chat/completions"
    model = os.getenv("IMAGE_GEN_MODEL", "gpt-image-2").strip()

    if not api_key:
        logger.error("IMAGE_GEN_API_KEY 未配置")
        return False

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "生成一张可爱的橘猫在窗台上晒太阳的图片，二次元插画风格",
            }
        ],
        "max_tokens": 4096,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info("=" * 60)
    logger.info("测试 2: /v1/chat/completions 端点（兼容模式）")
    logger.info("请求 payload: %s", json.dumps(payload, ensure_ascii=False))

    started = time.monotonic()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600, connect=30),
            ) as resp:
                elapsed = time.monotonic() - started
                body = await resp.text()

                logger.info("HTTP %d, 耗时 %.2fs, 响应大小 %d bytes", resp.status, elapsed, len(body))

                if resp.status != 200:
                    logger.error("❌ 请求失败: %s", body[:500])
                    return False

                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    logger.error("❌ 响应不是有效 JSON: %.500s", body)
                    return False

                logger.info("响应结构 keys: %s", list(data.keys()))
                logger.info("响应完整内容: %s", json.dumps(data, ensure_ascii=False, indent=2)[:2000])

                # 提取 content
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    logger.error("❌ 未找到 choices[0].message.content")
                    return False

                logger.info("content 类型: %s", type(content).__name__)
                if isinstance(content, list):
                    logger.info("content 是数组，包含 %d 个元素", len(content))
                    for i, item in enumerate(content):
                        logger.info("  [%d] type=%s, keys=%s", i, item.get("type"), list(item.keys()) if isinstance(item, dict) else "N/A")

                # 尝试提取图片 URL
                image_url = None
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url_obj = item.get("image_url", {})
                            if isinstance(image_url_obj, dict):
                                image_url = image_url_obj.get("url")
                                break
                elif isinstance(content, str):
                    # 可能需要在文本中提取 URL
                    import re
                    urls = re.findall(r"https?://[^\s)]+", content)
                    if urls:
                        image_url = urls[-1].rstrip(".,，。")

                if not image_url:
                    # 尝试从 data[0].url 提取
                    try:
                        image_url = data["data"][0]["url"]
                    except (KeyError, IndexError, TypeError):
                        pass

                if not image_url:
                    logger.warning("⚠️  未能从 chat completions 响应中提取图片 URL")
                    logger.info("这可能是预期的——chat completions 端点主要用于文本，生图请用 /v1/images/generations")
                    return False

                logger.info("✅ 提取到图片 URL: %s", image_url)

    except Exception as exc:
        logger.error("❌ 请求异常: %s", exc)
        return False

    # 下载图片
    logger.info("正在下载图片...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    logger.error("❌ 下载图片失败: HTTP %d", resp.status)
                    return False
                image_bytes = await resp.read()
                logger.info("✅ 下载成功: %d bytes", len(image_bytes))

                output_dir = project_root / "assets" / "generated_images"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"test_chat_{timestamp}.png"
                output_path.write_bytes(image_bytes)
                logger.info("✅ 图片已保存到: %s (%.1f KB)", output_path, len(image_bytes) / 1024)
    except Exception as exc:
        logger.error("❌ 下载图片异常: %s", exc)
        return False

    return True


async def main():
    logger.info("Right.Codes 生图 API 验证脚本")
    logger.info("API Key: %s...", os.getenv("IMAGE_GEN_API_KEY", "")[:20])
    logger.info("Model: %s", os.getenv("IMAGE_GEN_MODEL", "gpt-image-2"))
    logger.info("")

    results = {}

    # 测试 images 端点（推荐）
    results["images"] = await test_images_endpoint()

    # 测试 chat completions 端点（兼容性）
    results["chat"] = await test_chat_completions_endpoint()

    # 汇总
    logger.info("=" * 60)
    logger.info("测试结果汇总:")
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        logger.info("  %s: %s", name, status)

    all_pass = all(results.values())
    if all_pass:
        logger.info("\n🎉 所有测试通过！生图 API 工作正常。")
    else:
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        logger.warning("\n⚠️  %d/%d 测试通过", passed, total)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
