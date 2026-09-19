"""Protocol-preserving context projection backed by request-local evidence."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from shuiyuan_auto_reply.application.tool_results import current_turn

# Token budget for prior conversation turns and character budget for the
# recent-discussion block; both are fixed so the dynamic tool loop owns the rest.
HISTORY_TOKEN_BUDGET = 4000
RECENT_CHARS = 6000


def text_value(value) -> str:
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )


def compact_content(content, limit: int):
    text = text_value(content)
    if len(text) <= limit:
        return content
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        kept = []
        for item in payload["items"]:
            candidate = {
                **payload,
                "items": kept + [item],
                "truncated": True,
            }
            if len(json.dumps(candidate, ensure_ascii=False, default=str)) > limit:
                break
            kept.append(item)
        payload["items"] = kept
        payload["truncated"] = True
        return json.dumps(payload, ensure_ascii=False, default=str)
    return json.dumps(
        {
            "excerpt": text[:limit],
            "truncated": True,
            "total_chars": len(text),
        },
        ensure_ascii=False,
    )


def repair_tool_pairing(messages):
    """Return ``(messages, repaired_call_ids)`` with every tool call answered.

    OpenAI-compatible Responses endpoints reject an input where a function call
    has no matching output, or where an output appears before its call. A single
    broken pair therefore fails every later request in the turn. Repair the
    projection here so one bad pair costs one round instead of the whole answer.
    """
    seen: set[str] = set()
    kept: list = []
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in seen:
                continue
        else:
            for call in getattr(message, "tool_calls", None) or []:
                if call.get("id"):
                    seen.add(call["id"])
        kept.append(message)
    answered = {
        m.tool_call_id for m in kept if isinstance(m, ToolMessage) and m.tool_call_id
    }
    result: list = []
    repaired: list[str] = []
    for message in kept:
        result.append(message)
        for call in getattr(message, "tool_calls", None) or []:
            call_id = call.get("id")
            if call_id and call_id not in answered:
                repaired.append(call_id)
                result.append(
                    ToolMessage(
                        content=(
                            "Tool result unavailable in this context; call the tool "
                            "again if this result is still needed."
                        ),
                        tool_call_id=call_id,
                        name=call.get("name") or "",
                        status="error",
                    )
                )
    return result, repaired


def _group(messages):
    """Split into atomic units: a tool-calling AI message travels with its results."""
    groups: list[list] = []
    for message in messages:
        if (
            isinstance(message, ToolMessage)
            and groups
            and isinstance(groups[-1][0], AIMessage)
            and groups[-1][0].tool_calls
        ):
            groups[-1].append(message)
        else:
            groups.append([message])
    return groups


# Once over budget, drop enough of the oldest tool rounds to land well under it so
# the next several requests grow from a stable prefix instead of shifting again.
_DROP_TARGET = 0.6
_KEEP_LATEST_GROUPS = 3
_FALLBACK_TOOL_CHARS = 300


def project_messages(messages, budget: int = 60_000, *, preserve_first: bool = True):
    """Fit the tool loop into ``budget`` without rewriting what is kept.

    Retained messages are byte-identical to previous rounds so the provider's
    prefix cache keeps hitting; full evidence stays readable through the turn
    store for the finalizer.  Only when the newest rounds alone exceed the
    budget are their bodies shortened.
    """
    messages = list(messages)
    if count_tokens_approximately(messages) <= budget:
        return messages
    turn = current_turn.get()
    if turn is None:
        return messages  # Never discard evidence without a readable backing store.
    groups = _group(messages)
    tool_groups = [
        i
        for i, group in enumerate(groups)
        if isinstance(group[0], AIMessage) and group[0].tool_calls
    ]
    protected = set(tool_groups[-_KEEP_LATEST_GROUPS:])
    protected.update(
        i
        for i, group in enumerate(groups)
        if getattr(group[0], "name", None) == "target_post"
    )
    if preserve_first and groups and isinstance(groups[0][0], HumanMessage):
        protected.add(0)
    target = budget * _DROP_TARGET
    retained = list(groups)
    dropped = 0
    for index in tool_groups:
        if index in protected:
            continue
        if count_tokens_approximately([m for g in retained for m in g]) <= target:
            break
        group = groups[index]
        turn.save(
            [
                {
                    "type": m.type,
                    "content": m.content,
                    "calls": getattr(m, "tool_calls", []),
                }
                for m in group
            ]
        )
        retained.remove(group)
        dropped += 1
    result = [m for g in retained for m in g]
    # Loose text messages (not the request, target or a tool round) are rare
    # and carry no pairing constraint, so shorten them before any tool body.
    if count_tokens_approximately(result) > budget:
        loose = {
            id(m)
            for i, g in enumerate(retained)
            if i not in protected
            and not (isinstance(g[0], AIMessage) and g[0].tool_calls)
            for m in g
            if not isinstance(m, ToolMessage)
        }
        result = [
            (
                m.model_copy(update={"content": compact_content(m.content, 1800)})
                if id(m) in loose
                else m
            )
            for m in result
        ]
    # The newest rounds alone can exceed the budget; shrink bodies but keep
    # every tool_call_id answered so the request stays valid.
    if count_tokens_approximately(result) > budget:
        result = [
            (
                m.model_copy(
                    update={"content": compact_content(m.content, _FALLBACK_TOOL_CHARS)}
                )
                if isinstance(m, ToolMessage)
                else m
            )
            for m in result
        ]
    logging.info(
        "Context projection: estimated_tokens=%d -> %d budget=%d dropped_rounds=%d evidence=%d",
        count_tokens_approximately(messages),
        count_tokens_approximately(result),
        budget,
        dropped,
        len(turn.results),
    )
    return result
