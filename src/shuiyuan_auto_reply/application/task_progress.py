"""Request-local investigation state and source-grounded evidence identities."""

import json
from dataclasses import dataclass, field
from hashlib import sha256
from urllib.parse import urlsplit, urlunsplit


@dataclass
class TaskProgress:
    goal: str = ""
    topic_id: int | None = None
    searches: list[dict] = field(default_factory=list)
    phase: str = "investigate"


def source_records(value):
    """Yield structured sources, never count tool envelopes or index wrappers."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return
    if isinstance(value, list):
        for item in value:
            yield from source_records(item)
    elif isinstance(value, dict):
        if str(value.get("ref", "")).startswith(("forum:", "topic:")):
            yield str(value["ref"]), value
            return
        if value.get("post_id"):
            yield "post:" + str(value["post_id"]), value
            return
        if value.get("username") and (value.get("user_id") or value.get("id")):
            yield "user:" + str(value.get("user_id", value.get("id"))), value
            return
        if value.get("url") and (
            value.get("snippet")
            or value.get("content")
            or value.get("text")
            or value.get("title")
        ):
            parts = urlsplit(value["url"])
            url = urlunsplit(
                (
                    parts.scheme.lower(),
                    parts.netloc.lower(),
                    parts.path,
                    parts.query,
                    "",
                )
            )
            page_start = value.get("page_start")
            page_suffix = f":page:{page_start}" if page_start is not None else ""
            content_suffix = ":content:" + content_digest(value)[:16]
            yield "web:" + url + page_suffix + content_suffix, value
            return
        for key in ("posts", "results", "items", "users", "user", "output", "text"):
            if key in value:
                yield from source_records(value[key])


def content_digest(record: dict) -> str:
    content = {
        k: record[k]
        for k in (
            "content",
            "snippet",
            "text",
            "ref",
            "author",
            "username",
            "avatar",
            "image_urls",
            "reply_to_post_number",
        )
        if k in record
    }
    if not content:
        content = {"title": record.get("title", "")}
    return sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
