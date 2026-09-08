"""Composition-root package with lazy exports to avoid feature cycles."""

__all__ = ["AppSettings", "ApplicationContainer"]


def __getattr__(name: str):
    if name == "AppSettings":
        from .settings import AppSettings

        return AppSettings
    if name == "ApplicationContainer":
        from .container import ApplicationContainer

        return ApplicationContainer
    raise AttributeError(name)
