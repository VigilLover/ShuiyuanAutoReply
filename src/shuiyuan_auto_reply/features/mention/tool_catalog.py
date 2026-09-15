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
    "search_user_by_id": "users",
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


def legacy_forum_operations(names):
    """Preserve operation-level permissions while legacy tools are consolidated."""
    if names is None:
        return {}
    enabled = set(names)
    result = {}
    if "forum_search" not in enabled:
        operations = set()
        if {"search_posts", "search_posts_by_time"} & enabled:
            # Legacy post search also returned topic metadata and was the only
            # topic-discovery entry point.
            operations.update({"posts", "topics"})
        result["forum_search"] = operations
    if "forum_read" not in enabled:
        operations = set()
        if {
            "get_post",
            "read_tool_result",
            "inspect_images",
            "inspect_image",
        } & enabled:
            operations.add("exact")
        if "recent_posts" in enabled:
            operations.add("topic")
        result["forum_read"] = operations
    if "users" not in enabled:
        operations = set()
        if "get_user" in enabled:
            operations.add("username")
        if "get_users" in enabled:
            operations.add("usernames")
        if "search_user" in enabled:
            operations.add("query")
        if {"search_user", "search_user_by_id"} & enabled:
            operations.add("user_id")
        result["users"] = operations
    return result
