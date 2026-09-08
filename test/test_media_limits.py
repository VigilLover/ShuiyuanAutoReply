import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from shuiyuan_auto_reply.infrastructure.persistence.resources import normalize_image


def test_bytes_checked_before_image_decode():
    config = SimpleNamespace(section=lambda _: {"max_image_bytes": 3})
    with (
        patch(
            "shuiyuan_auto_reply.bootstrap.deployment.get_deployment",
            return_value=config,
        ),
        patch("PIL.Image.open") as decode,
    ):
        with pytest.raises(ValueError, match="byte limit"):
            normalize_image(b"abcd")
        decode.assert_not_called()


def test_pixel_limit_and_resizing():
    data = io.BytesIO()
    Image.new("RGB", (20, 10)).save(data, format="PNG")
    limits = dict(max_image_bytes=1024, max_pixels=100, max_long_edge=10)
    config = SimpleNamespace(section=lambda _: limits)
    with patch(
        "shuiyuan_auto_reply.bootstrap.deployment.get_deployment", return_value=config
    ):
        with pytest.raises(ValueError, match="pixel limit"):
            normalize_image(data.getvalue())
        limits["max_pixels"] = 200
        resized = normalize_image(data.getvalue())
        with Image.open(io.BytesIO(resized)) as image:
            assert image.size == (10, 5)
