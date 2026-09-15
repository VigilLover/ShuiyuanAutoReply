"""Canonical model-facing tool names and one-way legacy configuration aliases."""

FORUM_TOOL_NAMES = ("forum_search", "forum_read", "users")
MODEL_TOOL_NAMES = (
    *FORUM_TOOL_NAMES,
    "generate_image",
    "search_mention_memory",
    "manage_mention_memory",
)

LEGACY_TOOL_ALIASES = {
    "search_posts": "forum_search",
    "search_posts_by_time": "forum_search",
    "get_post": "forum_read",
    "recent_posts": "forum_read",
    "read_tool_result": "forum_read",
    "inspect_images": "forum_read",
    "inspect_image": "forum_read",
    "get_user": "users",
    "get_users": "users",
    "search_user": "users",
    "prepare_image_references": "generate_image",
}


def migrate_tool_names(names):
    """Map stored legacy names once while preserving order and unknown providers."""
    if names is None:
        return None
    result = []
    for name in names:
        mapped = LEGACY_TOOL_ALIASES.get(name, name)
        if mapped not in result:
            result.append(mapped)
    return result
