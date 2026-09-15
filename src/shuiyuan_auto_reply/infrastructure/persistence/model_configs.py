"""Stored model-endpoint configurations: secret naming and runtime resolution.

The settings UI keeps a library of chat (text model) and image endpoints there,
with their keys in the secret vault. Both kinds resolve differently on purpose:

* chat is pinned by the runtime profile, so the switch writes the profile draft
  and goes through the existing hot-swap path;
* image is read from the active configuration at call time, so switching one
  takes effect immediately and survives a restart.
"""

from typing import Any

SECRET_PREFIX = "model-config:"


def secret_name(config_id: str) -> str:
    return f"{SECRET_PREFIX}{config_id}"


async def active_image_endpoint(store, vault) -> dict[str, Any] | None:
    """The image endpoint the active stored configuration selects, if any."""
    if store is None:
        return None
    active = await store.active_model_config("image")
    if active is None:
        return None
    api_key = ""
    if vault is not None:
        api_key = (await vault.get(secret_name(active["id"]))) or ""
    return {
        "base_url": active["base_url"],
        "api_key": api_key,
        "model": active["model"],
    }


def image_endpoint_resolver(store, vault):
    """Callable attached to the state store for the image tool to consult."""

    async def resolve() -> dict[str, Any] | None:
        return await active_image_endpoint(store, vault)

    return resolve
