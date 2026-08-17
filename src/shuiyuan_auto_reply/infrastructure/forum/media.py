"""Forum-only publication of locally generated media."""

import io
from pathlib import Path

from PIL import Image

from shuiyuan_auto_reply.domain import ForumMediaRef, GeneratedImageArtifact


class ForumMediaUploader:
    def __init__(self, forum_model, state_store=None) -> None:
        self.forum_model = forum_model
        self.state_store = state_store

    async def upload(self, artifact: GeneratedImageArtifact) -> ForumMediaRef:
        image_bytes = Path(artifact.local_path).read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            converted = image.convert("RGB") if image.mode != "RGB" else image
            buffer = io.BytesIO()
            converted.save(buffer, format="JPEG", quality=95)
        response = await self.forum_model.upload_image(buffer.getvalue())
        if self.state_store is not None:
            await self.state_store.set_forum_short_path(artifact.artifact_id, response.short_path)
        return ForumMediaRef(response.short_path)
