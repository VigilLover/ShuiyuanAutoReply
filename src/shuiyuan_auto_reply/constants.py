import os
import re as _re
from importlib import resources


class Settings:

    @property
    def assets_directory(self) -> str:
        return str(resources.files("shuiyuan_auto_reply") / "assets")

    @property
    def auto_reply_tag(self) -> str:
        return "<!-- 来自小狼的自动回复 -->"

    @property
    def legacy_auto_reply_tag(self) -> str:
        return "<!-- 来自南瓜的自动回复 -->"

    @property
    def auto_reply_tag_pattern(self) -> _re.Pattern:
        return _re.compile(
            _re.escape(self.auto_reply_tag) + "|" + _re.escape(self.legacy_auto_reply_tag)
        )

    def contains_auto_reply_tag(self, text: str) -> bool:
        return bool(self.auto_reply_tag_pattern.search(text))

    def remove_auto_reply_tag(self, text: str) -> str:
        return self.auto_reply_tag_pattern.sub("", text)

    @property
    def embedding_model_name(self) -> str:
        return os.getenv("EMBEDDING_MODEL_NAME", "moka-ai/m3e-base")

    @property
    def embedding_cache_folder(self) -> str | None:
        return os.getenv("EMBEDDING_CACHE_FOLDER")

    @property
    def embedding_dims(self) -> int:
        value = os.getenv("EMBEDDING_DIMS")
        if value is None:
            raise ValueError("Please set the EMBEDDING_DIMS environment variable.")
        return int(value)


settings = Settings()
