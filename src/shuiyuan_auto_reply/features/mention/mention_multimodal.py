"""Shuiyuan image URL recognition and normalization shared by tools and vision."""

import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import urlparse

SHUIYUAN_HOSTS = {"shuiyuan.sjtu.edu.cn"}
UPLOAD_SHORT_PATH_PREFIX = "/uploads/short-url/"
USER_AVATAR_PATH_PREFIX = "/user_avatar/"
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
)

_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]]*]\(\s*(?P<url>[^)\s]+)(?:\s+['\"][^'\"]*['\"])?\s*\)"
)
_HTML_IMG_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(?P<quote>['\"]?)(?P<url>[^'\"\s>]+)(?P=quote)",
    re.IGNORECASE,
)
_RAW_UPLOAD_RE = re.compile(r"(?<![\w/])upload://[^\s<>)\"']+", re.IGNORECASE)
_RAW_SHORT_PATH_RE = re.compile(
    r"(?<![\w-])/uploads/short-url/[^\s<>)\"']+", re.IGNORECASE
)
_RAW_SHUIYUAN_URL_RE = re.compile(
    r"https?://shuiyuan\.sjtu\.edu\.cn/uploads/short-url/[^\s<>)\"']+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImageInspectResult:
    """Structured tool artifact listing image URLs the model may look at."""

    image_urls: list[str]
    source: str = "forum_read"
    description: str = ""


def _is_probable_image_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path if parsed.scheme in {"http", "https"} else url
    return path.lower().endswith(IMAGE_EXTENSIONS)


def _strip_url_wrapping(url: str) -> str:
    return unescape(url.strip().strip("<>").rstrip(".,;:"))


_BASE62_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Discourse stores originals as .../<sha1>.<ext> and optimized variants as
# .../<sha1>_<version>_<w>x<h>.<ext>; both map to one upload:// short URL.
_UPLOAD_SHA1_RE = re.compile(
    r"^(?P<sha1>[0-9a-f]{40})(?:_\d+_\d+x\d+)?\.(?P<ext>[a-z0-9]+)$", re.IGNORECASE
)


def upload_short_url_from_path(path: str) -> str | None:
    """Rebuild the upload:// short URL Discourse derives from an upload's sha1.

    ``/secure-uploads/...`` and ``/uploads/default/original/...`` links need a
    signed request the bot cannot make, but ``/uploads/short-url/<base62>.<ext>``
    only needs the login cookie.  Discourse computes the token as the base62
    encoding of the sha1, so the authenticated path is always reachable.
    """
    match = _UPLOAD_SHA1_RE.match(path.rsplit("/", 1)[-1])
    if match is None:
        return None
    number = int(match.group("sha1"), 16)
    token = ""
    while number:
        number, remainder = divmod(number, 62)
        token = _BASE62_ALPHABET[remainder] + token
    return f"upload://{token or '0'}.{match.group('ext').lower()}"


def normalize_shuiyuan_image_url(url: str) -> str | None:
    candidate = _strip_url_wrapping(url)
    if not candidate:
        return None

    if candidate.startswith("upload://"):
        return candidate if _is_probable_image_url(candidate) else None

    if candidate.startswith(UPLOAD_SHORT_PATH_PREFIX):
        normalized = "upload://" + candidate[len(UPLOAD_SHORT_PATH_PREFIX) :]
        return normalized if _is_probable_image_url(normalized) else None

    if candidate.startswith(USER_AVATAR_PATH_PREFIX):
        return candidate if _is_probable_image_url(candidate) else None

    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc.lower() not in SHUIYUAN_HOSTS:
            return None
        if parsed.path.startswith(UPLOAD_SHORT_PATH_PREFIX):
            filename = parsed.path[len(UPLOAD_SHORT_PATH_PREFIX) :]
            normalized = "upload://" + filename
            return normalized if _is_probable_image_url(normalized) else None
        if parsed.path.startswith(
            (
                "/uploads/original/",
                "/secure-uploads/",
                "/uploads/default/original/",
                "/uploads/default/optimized/",
            )
        ):
            if not _is_probable_image_url(candidate):
                return None
            return upload_short_url_from_path(parsed.path) or candidate
        if parsed.path.startswith(USER_AVATAR_PATH_PREFIX):
            return parsed.path if _is_probable_image_url(parsed.path) else None

    return None


def extract_image_urls(text: str | None) -> list[str]:
    if not text:
        return []

    candidates: list[str] = []
    for pattern in (
        _MARKDOWN_IMAGE_RE,
        _HTML_IMG_RE,
        _RAW_SHUIYUAN_URL_RE,
        _RAW_SHORT_PATH_RE,
        _RAW_UPLOAD_RE,
    ):
        for match in pattern.finditer(text):
            candidates.append(
                match.group("url") if "url" in match.groupdict() else match.group(0)
            )

    seen: set[str] = set()
    normalized_urls: list[str] = []
    for candidate in candidates:
        normalized = normalize_shuiyuan_image_url(candidate)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        normalized_urls.append(normalized)
    return normalized_urls
