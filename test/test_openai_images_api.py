#!/usr/bin/env python3
"""Live-check an OpenAI-compatible Images API configuration.

Reads IMAGE_GEN_API_KEY, IMAGE_GEN_API_URL, and IMAGE_GEN_MODEL from .env by
default. IMAGE_GEN_API_URL must be a base URL such as https://4router.net/v1.

Examples:
    uv run python scripts/test_openai_images_api.py
    uv run python scripts/test_openai_images_api.py --reference path/to/ref.png
    uv run python scripts/test_openai_images_api.py --operation edit --reference upload.png
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "generated_images"


def _redact_key(api_key: str) -> str:
    if len(api_key) <= 10:
        return "<set>"
    return f"{api_key[:6]}...{api_key[-4:]}"


def _endpoint(base_url: str, operation: str) -> str:
    if operation not in {"generation", "edit"}:
        raise ValueError(f"Unsupported operation: {operation}")
    suffix = "generations" if operation == "generation" else "edits"
    return f"{base_url.rstrip('/')}/images/{suffix}"


def _guess_extension(image_bytes: bytes, fallback: str = ".png") -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return ".gif"
    return fallback


def _encode_reference(reference: str) -> str:
    if reference.startswith("data:"):
        return reference
    if reference.startswith(("http://", "https://")):
        return reference

    path = Path(reference).expanduser()
    image_bytes = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _request_preview(payload: dict[str, Any]) -> dict[str, Any]:
    preview = dict(payload)
    if "images" in preview:
        preview["images"] = [
            {
                "image_url": (
                    image["image_url"][:80] + "..."
                    if isinstance(image.get("image_url"), str)
                    and len(image["image_url"]) > 80
                    else image.get("image_url")
                )
            }
            for image in preview["images"]
        ]
    return preview


async def _download_image(session: aiohttp.ClientSession, url: str, timeout: float) -> bytes:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
        body = await response.read()
        if response.status != 200:
            raise RuntimeError(f"image download HTTP {response.status}: {body[:300]!r}")
        return body


async def _extract_image_bytes(
    session: aiohttp.ClientSession,
    response_payload: dict[str, Any],
    timeout: float,
) -> bytes:
    try:
        image = response_payload["data"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("response missing data[0]") from exc
    if not isinstance(image, dict):
        raise RuntimeError("response data[0] is not an object")

    b64_json = image.get("b64_json")
    if isinstance(b64_json, str) and b64_json:
        return base64.b64decode(b64_json, validate=True)

    image_url = image.get("url")
    if isinstance(image_url, str) and image_url:
        if image_url.startswith("data:"):
            _, _, encoded = image_url.partition(",")
            return base64.b64decode(encoded, validate=True)
        return await _download_image(session, image_url, timeout)

    raise RuntimeError("response data[0] has neither b64_json nor url")


async def _post_images_request(args: argparse.Namespace) -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("IMAGE_GEN_API_KEY", "").strip()
    base_url = os.getenv("IMAGE_GEN_API_URL", "").strip()
    model = os.getenv("IMAGE_GEN_MODEL", "gpt-image-2").strip() or "gpt-image-2"

    if not api_key:
        print("ERROR: IMAGE_GEN_API_KEY is not set")
        return 2
    if not base_url:
        print("ERROR: IMAGE_GEN_API_URL is not set. Use a base URL such as https://4router.net/v1")
        return 2

    operation = args.operation
    if operation == "auto":
        operation = "edit" if args.reference else "generation"

    endpoint = _endpoint(base_url, operation)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": args.prompt,
        "size": args.size,
        "n": 1,
    }
    if operation == "edit":
        if not args.reference:
            print("ERROR: --reference is required for --operation edit")
            return 2
        payload["images"] = [{"image_url": _encode_reference(item)} for item in args.reference]

    print(f"Base URL: {base_url}")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {model}")
    print(f"API key: {_redact_key(api_key)}")
    print("Request payload:")
    print(json.dumps(_request_preview(payload), ensure_ascii=False, indent=2))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    started = time.monotonic()
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=args.timeout, connect=30),
            ) as response:
                text = await response.text()
                elapsed = time.monotonic() - started
                print(f"HTTP {response.status} in {elapsed:.2f}s, response bytes={len(text.encode('utf-8'))}")
                if response.status != 200:
                    print("Response body preview:")
                    print(text[:1200])
                    return 1
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    print(f"ERROR: response is not JSON: {exc}")
                    print(text[:1200])
                    return 1
        except Exception as exc:
            print(f"ERROR: request failed: {type(exc).__name__}: {exc}")
            return 1

        print(f"Response keys: {list(data.keys())}")
        first_image = data.get("data", [{}])[0] if isinstance(data.get("data"), list) and data.get("data") else {}
        print(f"data[0] keys: {list(first_image.keys()) if isinstance(first_image, dict) else type(first_image)}")

        try:
            image_bytes = await _extract_image_bytes(session, data, args.timeout)
        except Exception as exc:
            print(f"ERROR: could not extract image bytes: {type(exc).__name__}: {exc}")
            return 1

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = _guess_extension(image_bytes)
    output_path = output_dir / f"live_images_api_{operation}_{time.strftime('%Y%m%d_%H%M%S')}{extension}"
    output_path.write_bytes(image_bytes)
    print(f"Saved image: {output_path} ({len(image_bytes) / 1024:.1f} KB)")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        choices=("auto", "generation", "edit"),
        default="auto",
        help="auto uses edit when --reference is provided, otherwise generation",
    )
    parser.add_argument(
        "--prompt",
        default="A simple square test image of a cute orange cat sitting on a windowsill, clean anime illustration.",
        help="prompt to send to the image model",
    )
    parser.add_argument("--size", default="1024x1024", help="image size, for example 1024x1024")
    parser.add_argument(
        "--reference",
        action="append",
        help="reference image path, URL, or data URL; can be passed multiple times",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="request timeout seconds")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="where to save the returned image")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_post_images_request(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
