import argparse
import asyncio
import csv
import html
import json
import re
import sys
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shuiyuan_auto_reply.shuiyuan.constants import (
    base_url,
    get_topic_url,
    post_search_url,
)
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel


CSV_FIELDS = [
    "topic_title",
    "categories",
    "is_pm",
    "post_raw",
    "post_cooked",
    "like_count",
    "reply_count",
    "url",
    "created_at",
]

# Discourse user action types used by Shuiyuan:
# 4 = new topic, 5 = reply. The default therefore captures "historical posts".
DEFAULT_ACTION_TYPES = [4, 5]
PAGE_SIZE = 30


async def with_retries(
    label: str,
    factory: Any,
    attempts: int = 5,
    delay: float = 2.0,
):
    for attempt in range(1, attempts + 1):
        try:
            return await factory()
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"{label} 失败: {exc}，{delay:g} 秒后重试 ({attempt}/{attempts})")
            await asyncio.sleep(delay)


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def parse_date_bound(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    if not value:
        return None

    value = value.strip()
    if not value:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        day = datetime.strptime(value, "%Y-%m-%d").date()
        dt = datetime.combine(day, time.max if end_of_day else time.min)
    else:
        dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_topic_id(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value

    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)

    match = re.search(r"/t/(?:[^/]+/)?(\d+)(?=[/?#]|$)", value)
    if match:
        return int(match.group(1))

    raise ValueError(f"无法从话题参数中解析 topic id: {value}")


def parse_topic_ids(values: list[str] | str | int | None) -> list[int]:
    if values is None:
        return []
    if isinstance(values, int):
        return [values]
    if isinstance(values, str):
        values = [values]

    topic_ids: list[int] = []
    seen: set[int] = set()
    for value in values:
        for part in re.split(r"[\s,]+", value.strip()):
            topic_id = parse_topic_id(part)
            if topic_id is None or topic_id in seen:
                continue
            topic_ids.append(topic_id)
            seen.add(topic_id)
    return topic_ids


def normalize_datetime(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def in_time_window(
    created_at: str | None,
    since_dt: datetime | None,
    until_dt: datetime | None,
) -> bool:
    created_dt = parse_iso_datetime(created_at)
    if created_dt is None:
        return True

    created_dt = normalize_datetime(created_dt)
    since_dt = normalize_datetime(since_dt)
    until_dt = normalize_datetime(until_dt)
    if since_dt and created_dt < since_dt:
        return False
    if until_dt and created_dt > until_dt:
        return False
    return True


def date_after(dt: datetime) -> date:
    dt = normalize_datetime(dt)
    return dt.date() + timedelta(days=1)


def strip_signature(raw: str | None) -> str:
    if not raw:
        return ""

    cooked_signature = re.compile(
        r"<div\s+data-signature[^>]*>.*?</div>", re.DOTALL | re.IGNORECASE
    )
    raw = cooked_signature.sub("", raw)
    raw = re.sub(r"\[right\].*?\[/right\]", "", raw, flags=re.IGNORECASE | re.DOTALL)
    return raw.strip()


def cooked_emoji_to_text(html: str) -> str:
    def replace_emoji(match: re.Match[str]) -> str:
        attrs = match.group(0)
        title_match = re.search(r'title="([^"]+)"', attrs)
        alt_match = re.search(r'alt="([^"]+)"', attrs)
        value = (title_match or alt_match)
        if value is None:
            return ""

        emoji = value.group(1)
        if not emoji.startswith(":"):
            emoji = f":{emoji}:"
        return emoji

    return re.sub(r"<img\b[^>]*\bemoji[^>]*>", replace_emoji, html)


def cooked_images_to_markdown(html: str) -> str:
    lightbox_re = re.compile(
        r'<a\b[^>]*class="[^"]*\blightbox\b[^"]*"[^>]*>.*?</a>',
        re.DOTALL | re.IGNORECASE,
    )

    def replace_lightbox(match: re.Match[str]) -> str:
        block = match.group(0)
        short_match = re.search(
            r'data-download-href="/uploads/short-url/([^"?]+)',
            block,
        )
        if short_match is None:
            return ""

        title_match = re.search(r'title="([^"]*)"', block)
        alt_match = re.search(r'alt="([^"]*)"', block)
        width_match = re.search(r'\bwidth="(\d+)"', block)
        height_match = re.search(r'\bheight="(\d+)"', block)

        name = (
            (title_match or alt_match).group(1)
            if (title_match or alt_match)
            else "image"
        )
        size = ""
        if width_match and height_match:
            size = f"|{width_match.group(1)}x{height_match.group(1)}"
        return f"![{name}{size}](upload://{short_match.group(1)})"

    return lightbox_re.sub(replace_lightbox, html)


def cooked_to_plainish_raw(cooked: str | None) -> str:
    if not cooked:
        return ""

    text = strip_signature(cooked)
    text = cooked_images_to_markdown(text)
    text = cooked_emoji_to_text(text)
    text = re.sub(r"</p>\s*<p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</?(?:p|div|span|strong|em|blockquote|aside|header)[^>]*>",
        "",
        text,
    )
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def category_label(
    category_id: int | None,
    category_map: dict[int, dict[str, Any]],
) -> str:
    if category_id is None or category_id not in category_map:
        return "-"

    category = category_map[category_id]
    name = category.get("name") or str(category_id)
    parent_id = category.get("parent_category_id")
    if parent_id and parent_id in category_map:
        parent_name = category_map[parent_id].get("name") or str(parent_id)
        return f"{parent_name}|{name}"
    return name


def post_to_archive_row(
    post: dict[str, Any],
    topic_id: int,
    topic_metadata: dict[str, Any],
    category_map: dict[int, dict[str, Any]],
    fallback_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback_action = fallback_action or {}
    category_id = post.get("category_id", topic_metadata["category_id"])
    if category_id is None:
        category_id = fallback_action.get("category_id")

    raw = strip_signature(post.get("raw"))
    if not raw:
        raw = cooked_to_plainish_raw(post.get("cooked"))

    is_pm = "是" if topic_metadata["archetype"] == "private_message" else "否"
    return {
        "topic_title": topic_metadata["title"],
        "categories": category_label(category_id, category_map),
        "is_pm": is_pm,
        "post_raw": raw,
        "post_cooked": strip_signature(post.get("cooked")),
        "like_count": post.get("like_count", 0),
        "reply_count": post.get("reply_count", 0),
        "url": f"{base_url}/t/topic/{topic_id}/{post.get('post_number')}",
        "created_at": post.get("created_at") or fallback_action.get("created_at") or "",
    }


async def fetch_category_map(model: ShuiyuanModel) -> dict[int, dict[str, Any]]:
    async def request_categories():
        response = await model._rate_limited_request("get", f"{base_url}/site.json")
        if response.status != 200:
            raise RuntimeError(f"获取分类列表失败: {await response.text()}")
        return response

    response = await with_retries("获取分类列表", request_categories)
    data = await response.json()
    categories = data.get("categories", [])
    return {
        int(category["id"]): category
        for category in categories
        if category.get("id") is not None
    }


async def fetch_user_actions(
    model: ShuiyuanModel,
    username: str,
    action_types: list[int],
    since_dt: datetime | None,
    until_dt: datetime | None,
    max_pages: int | None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen_post_ids: set[int] = set()
    offset = 0
    page = 1

    while True:
        if max_pages is not None and page > max_pages:
            break

        page_obj = await with_retries(
            f"获取用户动作第 {page} 页",
            lambda: model.get_actions(username, action_types, offset=offset),
        )
        page_actions = [asdict(action) for action in page_obj.user_actions]
        if not page_actions:
            break

        kept_on_page = 0
        page_times: list[datetime] = []
        for action in page_actions:
            created_at = action.get("created_at")
            created_dt = parse_iso_datetime(created_at)
            if created_dt:
                page_times.append(created_dt)

            post_id = action.get("post_id")
            if post_id in seen_post_ids:
                continue
            if not in_time_window(created_at, since_dt, until_dt):
                continue

            if post_id is not None:
                seen_post_ids.add(post_id)
            actions.append(action)
            kept_on_page += 1

        print(
            f"第 {page} 页: 获取 {len(page_actions)} 条，窗口内新增 {kept_on_page} 条，累计 {len(actions)} 条"
        )

        if since_dt and page_times and min(page_times) < since_dt:
            print("达到开始时间阈值，停止翻页。")
            break

        offset += PAGE_SIZE
        page += 1

    return actions


async def fetch_topic_metadata(
    model: ShuiyuanModel,
    topic_id: int,
    fallback_action: dict[str, Any],
) -> dict[str, Any]:
    async def request_topic():
        response = await model._rate_limited_request(
            "get",
            f"{get_topic_url}/{topic_id}.json",
        )
        if response.status != 200:
            raise RuntimeError(f"获取 topic {topic_id} 元数据失败: {await response.text()}")
        return await response.json()

    topic = await with_retries(f"获取 topic {topic_id} 元数据", request_topic)
    return {
        "title": topic.get("title") or fallback_action.get("title") or "",
        "category_id": topic.get("category_id") or fallback_action.get("category_id"),
        "archetype": topic.get("archetype") or "regular",
    }


async def fetch_topic_posts(
    model: ShuiyuanModel,
    topic_id: int,
    post_ids: list[int],
) -> dict[str, Any]:
    async def request_posts():
        response = await model._rate_limited_request(
            "get",
            f"{get_topic_url}/{topic_id}/posts.json",
            params={"post_ids[]": post_ids, "include_raw": "true"},
        )
        if response.status != 200:
            raise RuntimeError(
                f"获取 topic {topic_id} 的帖子详情失败: {await response.text()}"
            )
        return await response.json()

    return await with_retries(
        f"获取 topic {topic_id} 的帖子详情",
        request_posts,
    )


async def build_archive_rows(
    model: ShuiyuanModel,
    actions: list[dict[str, Any]],
    category_map: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actions_by_topic: dict[int, list[dict[str, Any]]] = {}
    action_by_post_id: dict[int, dict[str, Any]] = {}
    for action in actions:
        topic_id = action.get("topic_id")
        post_id = action.get("post_id")
        if topic_id is None or post_id is None:
            continue

        actions_by_topic.setdefault(int(topic_id), []).append(action)
        action_by_post_id[int(post_id)] = action

    rows_by_post_id: dict[int, dict[str, Any]] = {}
    raw_topic_payloads: list[dict[str, Any]] = []
    total_topics = len(actions_by_topic)

    for index, (topic_id, topic_actions) in enumerate(actions_by_topic.items(), start=1):
        post_ids = [int(action["post_id"]) for action in topic_actions]
        print(f"读取详情 {index}/{total_topics}: topic {topic_id}, {len(post_ids)} 条")
        payload = await fetch_topic_posts(model, topic_id, post_ids)
        raw_topic_payloads.append(payload)

        metadata = await fetch_topic_metadata(model, topic_id, topic_actions[0])
        topic_title = metadata["title"]
        archetype = metadata["archetype"]
        topic_category_id = metadata["category_id"]

        posts = payload.get("post_stream", {}).get("posts", [])
        for post in posts:
            post_id = post.get("id")
            if post_id is None:
                continue

            action = action_by_post_id.get(int(post_id), {})
            category_id = post.get("category_id", topic_category_id)
            if category_id is None:
                category_id = action.get("category_id")

            post["category_id"] = category_id
            rows_by_post_id[int(post_id)] = post_to_archive_row(
                post,
                topic_id,
                {
                    "title": topic_title,
                    "category_id": topic_category_id,
                    "archetype": archetype,
                },
                category_map,
                action,
            )

    rows = []
    for action in actions:
        post_id = action.get("post_id")
        if post_id is None:
            continue
        row = rows_by_post_id.get(int(post_id))
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda row: row["created_at"])
    return rows, raw_topic_payloads


async def build_topic_archive_rows(
    model: ShuiyuanModel,
    username: str,
    topic_id: int,
    since_dt: datetime | None,
    until_dt: datetime | None,
    category_map: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    topic_metadata = await fetch_topic_metadata(model, topic_id, {})
    rows: list[dict[str, Any]] = []
    topic_payloads: list[dict[str, Any]] = []
    seen_post_ids: set[int] = set()
    username_key = username.casefold()
    search_before: date | None = date_after(until_dt) if until_dt else None

    while True:
        page = 1
        oldest_seen: datetime | None = None
        found_any_posts = False

        while True:
            query_parts = [f"topic:{topic_id}", f"@{username}"]
            if search_before is not None:
                query_parts.append(f"before:{search_before.isoformat()}")
            query_parts.append("order:latest")
            query = " ".join(query_parts)

            async def request_search():
                response = await model._rate_limited_request(
                    "get",
                    post_search_url,
                    params={"q": query, "page": page},
                )
                if response.status != 200:
                    raise RuntimeError(
                        f"搜索 topic {topic_id} 第 {page} 页失败: {await response.text()}"
                    )
                return await response.json()

            search_payload = await with_retries(
                f"搜索 topic {topic_id} 第 {page} 页",
                request_search,
            )
            search_posts = search_payload.get("posts", [])
            if not search_posts:
                rows.sort(key=lambda row: row["created_at"])
                return rows, topic_payloads

            found_any_posts = True

            for topic in search_payload.get("topics", []):
                if topic.get("id") == topic_id:
                    topic_metadata = {
                        "title": topic.get("title") or topic_metadata["title"],
                        "category_id": topic.get("category_id")
                        or topic_metadata["category_id"],
                        "archetype": topic.get("archetype")
                        or topic_metadata["archetype"],
                    }
                    break

            page_times: list[datetime] = []
            post_ids: list[int] = []
            for post in search_posts:
                created_dt = parse_iso_datetime(post.get("created_at"))
                if created_dt:
                    created_dt = normalize_datetime(created_dt)
                    page_times.append(created_dt)
                    if oldest_seen is None or created_dt < oldest_seen:
                        oldest_seen = created_dt

                post_id = post.get("id")
                if (
                    post.get("topic_id") != topic_id
                    or (post.get("username") or "").casefold() != username_key
                    or post_id is None
                    or post_id in seen_post_ids
                    or not in_time_window(post.get("created_at"), since_dt, until_dt)
                ):
                    continue

                seen_post_ids.add(post_id)
                post_ids.append(post_id)

            print(
                f"搜索 topic {topic_id} 第 {page} 页: 获取 {len(search_posts)} 条，窗口内新增 {len(post_ids)} 条"
            )

            if post_ids:
                payload = await fetch_topic_posts(model, topic_id, post_ids)
                topic_payloads.append(payload)

                posts = payload.get("post_stream", {}).get("posts", [])
                for post in posts:
                    if (post.get("username") or "").casefold() != username_key:
                        continue
                    if not in_time_window(post.get("created_at"), since_dt, until_dt):
                        continue

                    rows.append(
                        post_to_archive_row(
                            post,
                            topic_id,
                            topic_metadata,
                            category_map,
                        )
                    )

            grouped = search_payload.get("grouped_search_result") or {}
            if not grouped.get("more_full_page_results"):
                rows.sort(key=lambda row: row["created_at"])
                return rows, topic_payloads
            if since_dt and page_times and min(page_times) < since_dt:
                print(f"topic {topic_id}: 达到开始时间阈值，停止搜索。")
                rows.sort(key=lambda row: row["created_at"])
                return rows, topic_payloads

            page += 1
            if page > 10:
                break

        if not found_any_posts or oldest_seen is None:
            break

        next_search_before = date_after(oldest_seen)
        if next_search_before == search_before:
            next_search_before = normalize_datetime(oldest_seen).date()
            print(
                f"topic {topic_id}: 同一天搜索结果超过 500 条，切到 before:{next_search_before.isoformat()} 继续，可能略过该日更早结果。"
            )

        search_before = next_search_before

    rows.sort(key=lambda row: row["created_at"])
    return rows, topic_payloads


async def build_topics_archive_rows(
    model: ShuiyuanModel,
    username: str,
    topic_ids: list[int],
    since_dt: datetime | None,
    until_dt: datetime | None,
    category_map: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    topic_payloads: list[dict[str, Any]] = []

    for index, topic_id in enumerate(topic_ids, start=1):
        print(f"处理话题 {index}/{len(topic_ids)}: topic {topic_id}")
        topic_rows, payloads = await build_topic_archive_rows(
            model,
            username,
            topic_id,
            since_dt,
            until_dt,
            category_map,
        )
        print(f"topic {topic_id}: 找到 @{username} 发言 {len(topic_rows)} 条")
        rows.extend(topic_rows)
        topic_payloads.extend(payloads)

    rows.sort(key=lambda row: row["created_at"])
    return rows, topic_payloads


def write_archive(
    username: str,
    rows: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    topic_payloads: list[dict[str, Any]],
    output_root: Path,
    save_json: bool,
) -> Path:
    archive_dir = output_root / username
    archive_dir.mkdir(parents=True, exist_ok=True)

    csv_path = archive_dir / "user_archive.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if save_json:
        with (archive_dir / "user_actions.json").open("w", encoding="utf-8") as f:
            json.dump(actions, f, ensure_ascii=False, indent=2)
        with (archive_dir / "topic_payloads.json").open("w", encoding="utf-8") as f:
            json.dump(topic_payloads, f, ensure_ascii=False, indent=2)

    return csv_path


async def run(args: argparse.Namespace) -> Path:
    since_dt = parse_date_bound(args.since)
    until_dt = parse_date_bound(args.until, end_of_day=True)
    topic_ids = parse_topic_ids(args.topic_id)
    action_types = [int(item) for item in args.action_types.split(",") if item.strip()]

    ShuiyuanModel._request_interval = args.interval
    model = await ShuiyuanModel.create(args.cookies)
    try:
        category_map = await fetch_category_map(model)
        if topic_ids:
            actions = []
            rows, topic_payloads = await build_topics_archive_rows(
                model,
                args.username,
                topic_ids,
                since_dt,
                until_dt,
                category_map,
            )
        else:
            actions = await fetch_user_actions(
                model,
                args.username,
                action_types,
                since_dt,
                until_dt,
                args.max_pages,
            )
            rows, topic_payloads = await build_archive_rows(model, actions, category_map)

        return write_archive(
            args.username,
            rows,
            actions,
            topic_payloads,
            Path(args.output_root),
            args.save_json,
        )
    finally:
        await model.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Shuiyuan user history and save it as user_archive.csv."
    )
    parser.add_argument("username", nargs="?", help="水源用户名，例如 wolf_lumine")
    parser.add_argument("--since", help="开始时间，YYYY-MM-DD 或 ISO datetime")
    parser.add_argument("--until", help="结束时间，YYYY-MM-DD 或 ISO datetime")
    parser.add_argument(
        "--topic-id",
        action="append",
        help="只获取指定话题中此用户的发言；可重复传入，也可逗号分隔；留空则获取全站用户历史",
    )
    parser.add_argument("--max-pages", type=int, help="最多读取多少页 user_actions")
    parser.add_argument("--cookies", default=str(PROJECT_ROOT / "cookies"))
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "user_archive"),
        help="输出根目录，默认写入 user_archive/<username>/user_archive.csv",
    )
    parser.add_argument(
        "--action-types",
        default=",".join(map(str, DEFAULT_ACTION_TYPES)),
        help="Discourse user action type 列表，默认 4,5（主题+回复）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.3,
        help="请求间隔秒数，默认 0.3",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="同时保存 user_actions.json 和 topic_payloads.json 便于调试",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    interactive = not args.username
    if not args.username:
        args.username = input("请输入水源用户名: ").strip()
        args.since = input("请输入开始时间 (YYYY-MM-DD/ISO) 或留空: ").strip() or None
        args.until = input("请输入结束时间 (YYYY-MM-DD/ISO) 或留空: ").strip() or None
        args.topic_id = []
        while True:
            topic_value = input("请输入话题 ID/URL，留空结束: ").strip()
            if not topic_value:
                break
            args.topic_id.append(topic_value)

    if not args.username:
        parser.error("username 不能为空")
    topic_ids = parse_topic_ids(args.topic_id)
    if interactive and topic_ids:
        print(f"仅获取 topics {topic_ids} 中 @{args.username} 的发言。")

    csv_path = asyncio.run(run(args))
    print(f"已保存 {csv_path}")


if __name__ == "__main__":
    main()
