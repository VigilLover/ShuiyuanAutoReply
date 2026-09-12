import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from shuiyuan_auto_reply.bootstrap.deployment import DEFAULTS
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


@pytest.mark.parametrize("size", [(4032, 3024), (8000, 4000)])
def test_default_limits_resize_large_photos(size):
    data = io.BytesIO()
    with Image.new("RGB", size, "white") as image:
        image.save(data, format="JPEG")
    config = SimpleNamespace(section=lambda _: DEFAULTS["media"])
    with patch(
        "shuiyuan_auto_reply.bootstrap.deployment.get_deployment", return_value=config
    ):
        resized = normalize_image(data.getvalue())
    with Image.open(io.BytesIO(resized)) as image:
        image.load()
        assert max(image.size) == 2048
        assert image.format == "JPEG"
    assert len(resized) < DEFAULTS["media"]["max_image_bytes"]


def test_default_limits_reject_above_32_million_pixels():
    data = io.BytesIO()
    with Image.new("1", (8001, 4000)) as image:
        image.save(data, format="PNG")
    config = SimpleNamespace(section=lambda _: DEFAULTS["media"])
    with patch(
        "shuiyuan_auto_reply.bootstrap.deployment.get_deployment", return_value=config
    ):
        with pytest.raises(ValueError, match="pixel limit"):
            normalize_image(data.getvalue())
