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
        from shuiyuan_auto_reply.bootstrap.deployment import get_deployment
        return get_deployment().section("embedding")["model"]

    @property
    def embedding_cache_folder(self) -> str | None:
        from shuiyuan_auto_reply.bootstrap.deployment import get_deployment
        return get_deployment().section("embedding")["cache_folder"] or None

    @property
    def embedding_dims(self) -> int:
        from shuiyuan_auto_reply.bootstrap.deployment import get_deployment
        return get_deployment().section("embedding")["dims"]


settings = Settings()
