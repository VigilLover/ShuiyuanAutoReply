"""Request-local investigation state and source-grounded evidence identities."""

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from urllib.parse import urlsplit, urlunsplit


@dataclass
class TaskProgress:
    goal: str = ""
    topic_id: int | None = None
    authors: list[str] = field(default_factory=list)
    gaps: dict[str, str] = field(default_factory=dict)
    findings: list[dict] = field(default_factory=list)
    searches: list[dict] = field(default_factory=list)
    failed_directions: list[str] = field(default_factory=list)
    phase: str = "investigate"
    strategy: str = ""

    def view(self) -> dict:
        return asdict(self)

    def update(
        self,
        *,
        goal: str,
        gaps: dict[str, str],
        findings: list[dict],
        authors: list[str],
        evidence: dict,
        strategy: str = "",
    ) -> dict:
        if (
            len(goal) > 2000
            or len(gaps) > 12
            or len(findings) > 20
            or len(authors) > 50
            or len(strategy) > 1000
        ):
            raise ValueError("Task progress exceeds bounded state limits")
        if any(
            not key or len(key) > 80 or not value or len(value) > 500
            for key, value in gaps.items()
        ):
            raise ValueError("Gap IDs and descriptions must be bounded and nonempty")
        known_authors = {
            str(item.get("username", "")).casefold() for item in evidence.values()
        }
        if any(name.casefold() not in known_authors for name in authors):
            raise ValueError(
                "Target authors must be confirmed by source evidence, not the triggering user"
            )
        for finding in findings:
            ids = finding.get("evidence_ids", [])
            if (
                not ids
                or any(key not in evidence for key in ids)
                or len(str(finding.get("text", ""))) > 1000
            ):
                raise ValueError(
                    "Every finding needs existing evidence IDs and bounded text"
                )
            if authors and any(
                evidence[key].get("username")
                and evidence[key]["username"].casefold()
                not in {a.casefold() for a in authors}
                for key in ids
            ):
                raise ValueError("Finding evidence is outside selected author scope")
        self.goal, self.gaps, self.findings, self.authors, self.strategy = (
            goal,
            gaps,
            findings,
            authors,
            strategy,
        )
        return self.view()


def source_records(value):
    """Yield structured sources, never count tool envelopes or index wrappers."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return
    if isinstance(value, list):
        for item in value:
            yield from source_records(item)
    elif isinstance(value, dict):
        if value.get("post_id"):
            yield "post:" + str(value["post_id"]), value
            return
        if value.get("username") and (value.get("user_id") or value.get("id")):
            yield "user:" + str(value.get("user_id", value.get("id"))), value
            return
        if value.get("url") and (
            value.get("snippet") or value.get("content") or value.get("title")
        ):
            parts = urlsplit(value["url"])
            url = urlunsplit(
                (
                    parts.scheme.lower(),
                    parts.netloc.lower(),
                    parts.path,
                    parts.query,
                    "",
                )
            )
            yield "web:" + url, value
            return
        for key in ("posts", "results", "users", "user", "output", "text"):
            if key in value:
                yield from source_records(value[key])


def content_digest(record: dict) -> str:
    content = {
        k: v
        for k, v in record.items()
        if k not in {"result_id", "next_cursor", "read_full", "evidence_id", "warnings"}
    }
    return sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
