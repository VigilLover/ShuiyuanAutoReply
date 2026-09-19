"""Deterministic bounds for investigation; model prose cannot reset counters."""

import json
import time
from dataclasses import dataclass, field

READ_TOOLS = frozenset(
    {
        "forum_search",
        "forum_read",
        "users",
        "web_search",
        "web_read",
        "get_chuangka_menu",
    }
)
SEARCH_TOOLS = frozenset(
    {
        "forum_search",
        "web_search",
    }
)


def _image_ref(value: str) -> str:
    """``forum:42/7#image-1`` and ``42/7#image-1`` name the same image."""
    return value.strip().removeprefix("forum:")


def signature(name: str, args: dict) -> str:
    """Stable identity of a read so spelling variants replay instead of re-fetch."""
    args = {
        k: v
        for k, v in args.items()
        if k not in {"gap_id", "scope_reason"} and v is not None
    }
    for key in ("cursor", "page"):
        if args.get(key) == 0:
            args.pop(key)
    if args.get("refresh") is False:
        args.pop("refresh")
    for key in ("username", "term"):
        if isinstance(args.get(key), str):
            args[key] = args[key].strip().lstrip("@").casefold()
    if isinstance(args.get("usernames"), list):
        args["usernames"] = sorted(
            {str(item).strip().lstrip("@").casefold() for item in args["usernames"]}
        )
    if isinstance(args.get("image_refs"), list):
        args["image_refs"] = sorted({_image_ref(str(v)) for v in args["image_refs"]})
    return name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)


@dataclass
class RetrievalControl:
    model_rounds: int = 0
    queries: int = 0
    no_progress: int = 0
    repeats: int = 0
    stop_reason: str = ""
    seen: dict[str, object] = field(default_factory=dict)
    no_progress_batches: int = 2
    query_limit: int = 30
    model_limit: int = 14
    # Time kept for the final answer; an investigation call may not eat into it.
    final_reserve_seconds: int = 150
    # Hard cap for any single model request, investigation or final.
    model_call_timeout: int = 180

    def stop(self, progress, reason: str):
        progress.phase = "final"
        self.stop_reason = reason

    def before_model(self, progress, deadline: float):
        if self.model_rounds >= self.model_limit - 1:
            self.stop(progress, "model_budget")
        if self.queries >= self.query_limit:
            self.stop(progress, "query_budget")
        if time.monotonic() >= deadline - self.final_reserve_seconds:
            self.stop(progress, "time_budget")
        self.model_rounds += 1

    def call_timeout(self, deadline: float, *, final: bool) -> float:
        """Seconds one model request may take without starving the final answer."""
        remaining = deadline - time.monotonic()
        if not final:
            remaining -= self.final_reserve_seconds
        return max(0.1, min(remaining, self.model_call_timeout))

    def after_batch(self, progress, *, new_evidence: int, reads: int):
        if not reads or progress.phase == "final":
            return
        self.no_progress = 0 if new_evidence else self.no_progress + 1
        if self.no_progress >= self.no_progress_batches:
            self.stop(progress, "no_new_evidence")

    def metrics(self):
        return {
            k: getattr(self, k)
            for k in (
                "model_rounds",
                "queries",
                "no_progress",
                "repeats",
                "stop_reason",
            )
        }
