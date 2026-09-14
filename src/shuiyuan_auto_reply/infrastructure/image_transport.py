"""Safe URL quoting and bounded metadata for image download failures."""

from urllib.parse import quote

from yarl import URL


class ImageDownloadError(Exception):
    def __init__(self, status: int, retry_after: str | None = None):
        self.status = status
        self.retry_after = retry_after
        self.retryable = status in {408, 429} or status >= 500
        super().__init__(f"Image download HTTP {status}")


def encoded_image_url(url: str) -> URL:
    # Preserve existing escapes and signed query bytes; encode only unsafe/unicode chars.
    return URL(quote(url, safe=":/?#[]@!$&'()*+,;=%"), encoded=True)


def cached_media_attempt(func):
    """Share terminal download failures and successful preparation within a turn."""
    import inspect
    from functools import wraps

    @wraps(func)
    async def wrapped(*args, **kwargs):
        from shuiyuan_auto_reply.application.tool_results import current_turn

        turn = current_turn.get()
        if turn is None:
            return await func(*args, **kwargs)
        bound = inspect.signature(func).bind(*args, **kwargs)
        url = str(bound.arguments.get("url", ""))
        key = (
            str(encoded_image_url(url))
            if url.startswith(("http://", "https://"))
            else url
        )
        if key in turn.image_failures:
            raise turn.image_failures[key]
        cache_key = "prepared_media:" + func.__name__ + ":" + key
        if cache_key in turn.cache:
            return turn.cache[cache_key]
        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            status = getattr(exc, "status", getattr(exc, "status_code", None))
            response = getattr(exc, "response", None)
            status = status or getattr(response, "status_code", None)
            if getattr(exc, "retryable", None) is False or status in {
                400,
                401,
                403,
                404,
                410,
                422,
            }:
                turn.image_failures[key] = exc
            raise
        if result is not None:
            if (
                func.__name__ == "_download_and_encode"
                and isinstance(result, str)
                and result.startswith("data:")
            ):
                import base64

                remember_media(url, base64.b64decode(result.split(",", 1)[1]))
            turn.cache[cache_key] = result
        return result

    return wrapped


def media_key(url: str) -> str:
    return (
        str(encoded_image_url(url)) if url.startswith(("http://", "https://")) else url
    )


def remembered_media(url: str):
    from shuiyuan_auto_reply.application.tool_results import current_turn

    turn = current_turn.get()
    return turn.media_bytes.get(media_key(url)) if turn else None


def remember_media(url: str, data: bytes) -> None:
    from hashlib import sha256

    from shuiyuan_auto_reply.application.tool_results import current_turn
    from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

    turn = current_turn.get()
    if turn is None:
        return
    digest = sha256(data).hexdigest()
    config = get_deployment().section("media")
    if len(data) > config["max_image_bytes"]:
        raise ValueError("Image exceeds per-image byte budget")
    if digest not in turn.media_digests:
        if (
            len(turn.media_digests) >= config["max_images"]
            or sum(turn.media_digests.values()) + len(data) > config["max_turn_bytes"]
        ):
            raise ValueError("Image exceeds per-turn media budget")
        turn.media_digests[digest] = len(data)
    turn.media_bytes[media_key(url)] = data
