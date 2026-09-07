import asyncio
import json

import pytest

from shuiyuan_auto_reply.infrastructure.operations.migration import (
    digest,
    export_corpus,
    write_json,
)


def test_export_excludes_bot_and_signature(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "post_raw\nhello<div data-signature>signature</div>\n<!-- 来自小狼的自动回复 -->\n"
    )
    output = tmp_path / "out.jsonl"
    assert asyncio.run(export_corpus(output, csv_path=source, persona="wolf")) == 1
    assert json.loads(output.read_text()) == {"persona_id": "wolf", "text": "hello"}
    with pytest.raises(FileExistsError):
        asyncio.run(export_corpus(output, csv_path=source, persona="wolf"))


def test_checkpoint_atomic_and_fingerprint(tmp_path):
    path = tmp_path / "checkpoint"
    write_json(path, {"completed": [1]})
    before = digest(path)
    write_json(path, {"completed": [1, 2]})
    assert digest(path) != before
    assert not path.with_suffix(".tmp").exists()
