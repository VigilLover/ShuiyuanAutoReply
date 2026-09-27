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

    # The user asked for an image and generate_image is available.
    image_requested: bool = False
    image_attempted: bool = False
    # Model round in which one extra round was granted to generate the image.
    image_nudge_round: int | None = None

    @property
    def image_nudged(self) -> bool:
        return self.image_nudge_round is not None

    def stop(self, progress, reason: str):
        progress.phase = "final"
        self.stop_reason = reason

    def stop_investigation(self, progress, reason: str):
        """Stop for exhausted evidence, unless a requested image is still owed.

        The final phase cannot call tools, so ending there before generate_image
        ran leaves the model to invent an image. Grant one generation round.
        """
        if self.image_requested and not self.image_attempted:
            if self.image_nudge_round is None:
                self.image_nudge_round = self.model_rounds
                self.no_progress = 0
                return
            if self.image_nudge_round == self.model_rounds:
                # Both evidence checks can fire after the same batch.
                return
        self.stop(progress, reason)

    def before_model(self, progress, deadline: float):
        if self.model_rounds >= self.model_limit - 1:
            self.stop(progress, "model_budget")
        if self.queries >= self.query_limit:
            self.stop(progress, "query_budget")
        if time.monotonic() >= deadline - self.final_reserve_seconds:
            self.stop(progress, "time_budget")
        self.model_rounds += 1

    def call_timeout(
        self, deadline: float, *, final: bool, max_seconds: float | None = None
    ) -> float:
        """Bound one model or tool call without starving the final answer."""
        remaining = deadline - time.monotonic()
        if not final:
            remaining -= self.final_reserve_seconds
        timeout_cap = self.model_call_timeout if max_seconds is None else max_seconds
        return max(0.1, min(remaining, timeout_cap))

    def after_batch(self, progress, *, new_evidence: int, reads: int):
        if not reads or progress.phase == "final":
            return
        self.no_progress = 0 if new_evidence else self.no_progress + 1
        if self.no_progress >= self.no_progress_batches:
            self.stop_investigation(progress, "no_new_evidence")

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
