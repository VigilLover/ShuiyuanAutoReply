__all__ = ["MentionModel"]


def __getattr__(name: str):
    if name == "MentionModel":
        from .mention_model import MentionModel

        return MentionModel
    raise AttributeError(name)
