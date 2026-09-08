"""Bounded local media work and retention for small hosts."""

import asyncio
import io
import time
import weakref
from functools import wraps

_locks = weakref.WeakKeyDictionary()


def media_lock():
    loop = asyncio.get_running_loop()
    return _locks.setdefault(loop, asyncio.Semaphore(1))


def normalize_image(data):
    from PIL import Image

    from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

    config = get_deployment().section("media")
    if len(data) > config["max_image_bytes"]:
        raise ValueError("Image exceeds upload byte limit")
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > config["max_pixels"]:
            raise ValueError("Image exceeds decoded pixel limit")
        if max(image.size) <= config["max_long_edge"]:
            return data
        # Large animated uploads are flattened to one frame under the low-resource policy.
        image.thumbnail((config["max_long_edge"], config["max_long_edge"]))
        out = io.BytesIO()
        image.convert("RGB").save(out, format="JPEG", quality=85)
        return out.getvalue()


def bounded_media(function):
    @wraps(function)
    async def wrapper(*args, **kwargs):
        async with media_lock():
            data = kwargs.get("data")
            positional = list(args)
            if (
                data is None
                and len(positional) > 1
                and isinstance(positional[1], bytes)
            ):
                data = positional[1]
            if data is not None:
                from shuiyuan_auto_reply.features.mention.deepseek_vision import (
                    VisionMediaError,
                )

                try:
                    data = await asyncio.to_thread(normalize_image, data)
                    await asyncio.to_thread(enforce_quota, len(data))
                except (ValueError, OSError) as exc:
                    raise VisionMediaError(
                        "无法识别图片内容" if isinstance(exc, OSError) else str(exc)
                    ) from exc
                if "data" in kwargs:
                    kwargs["data"] = data
                else:
                    positional[1] = data
            return await function(*positional, **kwargs)

    return wrapper


def enforce_quota(incoming=0):
    from shuiyuan_auto_reply.bootstrap.deployment import get_deployment
    from shuiyuan_auto_reply.infrastructure.persistence.state import state_directory

    config = get_deployment().section("media")
    root = state_directory() / "artifacts"
    if not root.exists():
        return
    now = time.time()
    size = 0
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            if now - stat.st_mtime > config["retention_days"] * 86400:
                path.unlink(missing_ok=True)
            else:
                size += stat.st_size
    if size + incoming > config["quota_bytes"]:
        raise ValueError("Artifact storage quota exceeded; clear old conversations")
