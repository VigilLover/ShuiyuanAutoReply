"""Lossless raw cleanup and metadata extraction for forum tool results."""

import re
from urllib.parse import unquote

from bs4 import BeautifulSoup
from markdownify import markdownify


def clean_raw(text: str) -> str:
    # Only remove decorations at the end, never quoted examples or code blocks.
    text = re.sub(
        r"(?:\s*<!--\s*(?:[A-Za-z0-9]{20}|来自[^\n<>]*的自动回复)\s*-->)+\s*$", "", text
    )
    for match in reversed(
        list(re.finditer(r'(?m)^<div\s+data-signature(?:="[^"]*")?\s*>', text))
    ):
        if (
            re.search(r"</div>\s*$", text[match.start() :])
            and len(re.findall(r"(?m)^\s*```", text[: match.start()])) % 2 == 0
        ):
            text = text[: match.start()]
            break
    return text.strip()


def parse_content(raw: str | None, cooked: str) -> dict:
    soup = BeautifulSoup(cooked or "", "html.parser")
    mentions, links = [], []
    seen = set()
    for tag in soup.select("a.mention[href]"):
        if tag.find_parent(["pre", "code"]):
            continue
        href = tag.get("href", "")
        if "/u/" not in href:
            continue
        username = unquote(href.split("/u/", 1)[1].split("/")[0])
        quoted = tag.find_parent(["blockquote", "aside"]) is not None
        key = (username.casefold(), quoted)
        if key not in seen:
            seen.add(key)
            mentions.append(
                {"username": username, "source": "quote" if quoted else "body"}
            )
    for tag in soup.select("a[href]"):
        href = tag["href"]
        if href not in links:
            links.append(href)
    for tag in soup.select("script, style, [data-signature]"):
        tag.decompose()
    for tag in soup.select("img.emoji"):
        tag.replace_with(tag.get("alt", ""))
    content = clean_raw(raw) if raw is not None else markdownify(str(soup)).strip()
    if not mentions and raw:
        plain = re.sub(r"```.*?```|`[^`]*`", "", raw, flags=re.S)
        for line in plain.splitlines():
            quoted = line.lstrip().startswith(">") or "[quote" in line
            for username in re.findall(r"(?<![\w@])@([^\s@<>\[\]，。！？,:;]+)", line):
                key = (username.casefold(), quoted)
                if key not in seen:
                    seen.add(key)
                    mentions.append(
                        {"username": username, "source": "quote" if quoted else "body"}
                    )
    return {
        "content": content,
        "content_source": "raw" if raw is not None else "cooked_text",
        "mentions": mentions,
        "links": links,
    }
