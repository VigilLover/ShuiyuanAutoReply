"""Protocol-preserving context projection backed by request-local evidence."""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from shuiyuan_auto_reply.application.tool_results import current_turn


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


def project_messages(messages, budget: int = 24_000, *, preserve_first: bool = True):
    """Keep call/result pairing, save full evidence before shortening the projection."""
    messages = list(messages)
    if count_tokens_approximately(messages) <= budget:
        return messages
    turn = current_turn.get()
    if turn is None:
        return messages  # Never discard evidence without a readable backing store.
    # The caller's original messages are immutable evidence, never edit in place.
    projected = []
    for index, message in enumerate(messages):
        # Keep current request (first human) intact. Tool payloads are independently readable.
        if (
            preserve_first and index == 0 and isinstance(message, HumanMessage)
        ) or getattr(message, "name", None) == "target_post":
            projected.append(message)
        else:
            content = compact_content(message.content, 1800)
            projected.append(message.model_copy(update={"content": content}))
    if count_tokens_approximately(projected) <= budget:
        return projected
    groups = []
    for message in projected:
        if (
            isinstance(message, ToolMessage)
            and groups
            and isinstance(groups[-1][0], AIMessage)
            and groups[-1][0].tool_calls
        ):
            groups[-1].append(message)
        else:
            groups.append([message])
    # Preserve target reference and the newest two atomic tool interaction blocks.
    tool_groups = [
        i
        for i, group in enumerate(groups)
        if isinstance(group[0], AIMessage) and group[0].tool_calls
    ]
    protected = set(tool_groups[-2:])
    protected.update(
        i
        for i, group in enumerate(groups)
        if getattr(group[0], "name", None) == "target_post"
    )
    if preserve_first:
        protected.add(0)
    retained = []
    for i, group in enumerate(groups):
        if (
            i not in protected
            and count_tokens_approximately(
                [m for g in groups[i:] for m in g] + [m for g in retained for m in g]
            )
            > budget - 1500
        ):
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
        else:
            retained.append(group)
    groups = retained
    result = [m for g in groups for m in g]
    # Huge latest batches keep every tool_call_id, shrinking bodies rather than deleting responses.
    if count_tokens_approximately(result) > budget:
        result = [
            (
                m.model_copy(update={"content": compact_content(m.content, 300)})
                if isinstance(m, ToolMessage)
                else m
            )
            for m in result
        ]
    logging.info(
        "Context projection: estimated_tokens=%d -> %d budget=%d evidence=%d",
        count_tokens_approximately(messages),
        count_tokens_approximately(result),
        budget,
        len(turn.results),
    )
    return result
