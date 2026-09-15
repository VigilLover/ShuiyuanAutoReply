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
    }
)
SEARCH_TOOLS = frozenset(
    {
        "forum_search",
        "web_search",
    }
)


def signature(name: str, args: dict) -> str:
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
            args[key] = args[key].strip().casefold()
    return name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)


@dataclass
class RetrievalControl:
    model_rounds: int = 0
    queries: int = 0
    no_progress: int = 0
    repeats: int = 0
    stop_reason: str = ""
    seen: dict[str, object] = field(default_factory=dict)
    no_progress_batches: int = 3
    query_limit: int = 40
    model_limit: int = 24
    final_reserve_seconds: int = 60

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
