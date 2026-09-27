"""Cheap detection of image-generation requests in the user's post."""

import re

_IMAGE_REQUEST_RE = re.compile(
    r"生图|[Pp]图|画(?:一|个|张|幅|只|下|出|成)|"
    r"(?:生成|绘制|画|做|改|修改|加工|重做|换|加)"
    r".{0,12}?(?:图|头像|背景|壁纸|插画|表情包|立绘)"
)


def wants_image(text: str) -> bool:
    return bool(_IMAGE_REQUEST_RE.search(str(text or "")))
