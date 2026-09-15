"""Deterministic bounds for investigation; model prose cannot reset counters."""

import json
import time
from dataclasses import dataclass, field

READ_TOOLS = frozenset(
    {
        "get_post",
        "get_user",
        "search_user",
        "search_posts",
        "recent_posts",
        "search_posts_by_time",
        "web_search",
        "fetch_webpage_content",
        "image_search",
        "read_tool_result",
    }
)
SEARCH_TOOLS = frozenset(
    {
        "search_posts",
        "recent_posts",
        "search_posts_by_time",
        "web_search",
        "image_search",
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
    reviews: int = 0
    continuations: int = 0
    continuation_batches: int = 0
    review_attempts: int = 0
    stop_reason: str = ""
    seen: dict[str, object] = field(default_factory=dict)
    strategies: set[str] = field(default_factory=set)
    no_progress_batches: int = 3
    continuation_limit: int = 1
    continuation_batch_limit: int = 2
    query_limit: int = 40
    model_limit: int = 24
    final_reserve_seconds: int = 60

    def stop(self, progress, reason: str):
        progress.phase = "final"
        self.stop_reason = reason

    def review(self, progress, reason: str):
        if progress.phase == "final":
            return
        if self.continuations >= self.continuation_limit:
            self.stop(progress, reason)
            return
        if progress.phase != "review":
            self.reviews += 1
            self.review_attempts = 0
            progress.phase = "review"
            progress.failed_directions.append(reason)
            progress.failed_directions = progress.failed_directions[-12:]
            if progress.strategy:
                self.strategies.add(progress.strategy.strip().casefold())

    def before_model(self, progress, deadline: float):
        if self.model_rounds >= self.model_limit - 1:
            self.stop(progress, "model_budget")
        if self.queries >= self.query_limit:
            self.stop(progress, "query_budget")
        if time.monotonic() >= deadline - self.final_reserve_seconds:
            self.stop(progress, "time_budget")
        if progress.phase == "review" and self.review_attempts >= 2:
            self.stop(progress, "review_not_resolved")
        if progress.phase == "review":
            self.review_attempts += 1
        self.model_rounds += 1

    def continue_after_review(self, progress):
        strategy = progress.strategy.strip().casefold()
        if not strategy or strategy in self.strategies or not progress.gaps:
            raise ValueError(
                "Review needs unresolved gap IDs and a different concrete strategy, or answer now"
            )
        self.strategies.add(strategy)
        self.continuations += 1
        self.continuation_batches = 0
        progress.phase = "continue"

    def after_batch(self, progress, *, new_evidence: int, reads: int):
        if not reads or progress.phase in {"review", "final"}:
            return
        self.no_progress = 0 if new_evidence else self.no_progress + 1
        if progress.phase == "continue":
            self.continuation_batches += 1
            if self.continuation_batches >= self.continuation_batch_limit:
                self.stop(progress, "continuation_complete")
        if self.no_progress >= self.no_progress_batches:
            self.review(progress, "no_new_evidence")

    def metrics(self):
        return {
            k: getattr(self, k)
            for k in (
                "model_rounds",
                "queries",
                "no_progress",
                "repeats",
                "reviews",
                "continuations",
                "stop_reason",
            )
        }
