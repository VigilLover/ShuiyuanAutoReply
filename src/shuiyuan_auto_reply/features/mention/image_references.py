"""Generic, request-local reference media preparation and partial failure reporting."""

import asyncio
import base64
import io
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from uuid import uuid4

import aiohttp
from PIL import Image

from shuiyuan_auto_reply.application.events import emit_event
from shuiyuan_auto_reply.application.tool_results import cached_query, current_turn
from shuiyuan_auto_reply.infrastructure.image_transport import (
    ImageDownloadError,
    encoded_image_url,
)


def _retry_delay(exc, attempt):
    value = getattr(exc, "retry_after", None)
    if value:
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(
                    0,
                    (
                        parsedate_to_datetime(value) - datetime.now(timezone.utc)
                    ).total_seconds(),
                )
            except (ValueError, TypeError):
                pass
    return float(2**attempt)


async def prepare_references(
    references: list[dict[str, str]], *, model, strict_remote: bool = True
) -> dict:
    from .image_generation import _MAX_TOTAL_REFERENCE_BYTES, _download_and_encode
    from .shuiyuan_tools_objects import UserShort

    turn = current_turn.get()
    if not references or len(references) > 50:
        return {
            "status": "error",
            "error": "Provide 1 to 50 references",
            "items": [],
            "data_urls": [],
        }
    keys = set()
    for item in references:
        if (
            not isinstance(item, dict)
            or not item.get("key")
            or not item.get("url")
            or item["key"] in keys
        ):
            return {
                "status": "error",
                "error": "Every reference needs a unique key and a URL",
                "items": [],
                "data_urls": [],
            }
        keys.add(item["key"])
    gate = asyncio.Semaphore(4)
    deadline = turn.deadline if turn else time.monotonic() + 120

    async with aiohttp.ClientSession() as session:

        async def load(item):
            url = item["url"]

            async def fetch():
                nonlocal url
                refreshed = False
                async with gate:
                    for attempt in range(3):
                        try:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError(
                                    "Reference preparation deadline exceeded"
                                )
                            async with asyncio.timeout(remaining):
                                data = await _download_and_encode(
                                    session,
                                    url,
                                    shuiyuan_model=model,
                                    strict_remote=strict_remote,
                                    raise_errors=True,
                                )
                            if not data:
                                return {
                                    "status": "error",
                                    "error": "invalid_or_unavailable_image",
                                    "retryable": False,
                                }
                            with Image.open(
                                io.BytesIO(base64.b64decode(data.split(",", 1)[1]))
                            ) as image:
                                image.verify()
                            return {"status": "ok", "data": data, "url": url}
                        except Exception as exc:
                            if (
                                isinstance(exc, ImageDownloadError)
                                and exc.status == 404
                                and turn
                                and not refreshed
                            ):
                                refreshed = True
                                username = next(
                                    (
                                        value.get("username")
                                        for key, value in turn.cache.items()
                                        if key.startswith("user:")
                                        and isinstance(value, dict)
                                        and value.get("avatar")
                                        and encoded_image_url(value["avatar"])
                                        == encoded_image_url(url)
                                    ),
                                    None,
                                )
                                if username:
                                    user = await model.get_user_by_username(username)
                                    avatar = (
                                        UserShort(user, include_avatar=True).avatar
                                        if user
                                        else None
                                    )
                                    if avatar and encoded_image_url(
                                        avatar
                                    ) != encoded_image_url(url):
                                        url = avatar
                                        continue
                            retryable = isinstance(
                                exc, (TimeoutError, aiohttp.ClientConnectionError)
                            ) or getattr(exc, "retryable", False)
                            delay = _retry_delay(exc, attempt)
                            if (
                                retryable
                                and attempt < 2
                                and time.monotonic() + delay < deadline
                            ):
                                await asyncio.sleep(delay)
                                continue
                            return {
                                "status": "error",
                                "error": str(exc)[:200],
                                "retryable": retryable,
                            }
                    return {
                        "status": "error",
                        "error": "reference_attempts_exhausted",
                        "retryable": False,
                    }

            key = (
                "image:" + str(encoded_image_url(url))
                if url.startswith(("http://", "https://"))
                else "image:" + url
            )
            return await cached_query(key, fetch)

        loaded = await asyncio.gather(*(load(item) for item in references))
    items, data_urls, total = [], [], 0
    for source, result in zip(references, loaded):
        item = {
            "key": source["key"],
            "label": source.get("label") or source["key"],
            "url": result.get("url", source["url"]),
            "status": result["status"],
        }
        if result["status"] == "ok":
            size = len(base64.b64decode(result["data"].split(",", 1)[1]))
            if total + size <= _MAX_TOTAL_REFERENCE_BYTES:
                total += size
                data_urls.append(result["data"])
                item["index"] = len(data_urls)
            else:
                item.update(
                    status="error",
                    error="total_reference_bytes_exceeded",
                    retryable=False,
                )
        else:
            item.update(error=result["error"], retryable=result.get("retryable", False))
        items.append(item)
    prepared = {
        "reference_set_id": str(uuid4()),
        "status": (
            "ok"
            if len(data_urls) == len(items)
            else "partial" if data_urls else "error"
        ),
        "items": items,
        "successful": len(data_urls),
        "failed": len(items) - len(data_urls),
        "data_urls": data_urls,
    }
    if turn:
        turn.references[prepared["reference_set_id"]] = prepared
    await emit_event(
        "image.references_prepared",
        {"successful": len(data_urls), "failed": len(items) - len(data_urls)},
    )
    return prepared
