"""Serve the built Vue console when its static bundle is present.

Not an APIRouter: registration is conditional on the bundle existing (a bare
API-only deployment need not carry the frontend), so this mounts directly on
the app instance instead of being included unconditionally.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_static(api: FastAPI) -> None:
    static_dir = Path(__file__).parents[1] / "static"
    if not static_dir.is_dir():
        return

    assets_dir = static_dir / "assets"
    if assets_dir.is_dir():
        api.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @api.get("/favicon.ico", include_in_schema=False)
    async def frontend_favicon():
        return FileResponse(
            static_dir / "assets" / "favicon.ico",
            media_type="image/x-icon",
        )

    @api.get("/apple-touch-icon.png", include_in_schema=False)
    async def frontend_apple_touch_icon():
        return FileResponse(
            static_dir / "assets" / "apple-touch-icon.png",
            media_type="image/png",
        )

    @api.get("/", include_in_schema=False)
    async def frontend_index():
        return FileResponse(static_dir / "index.html")

    @api.get("/{frontend_path:path}", include_in_schema=False)
    async def frontend_history_fallback(frontend_path: str):
        if frontend_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(static_dir / "index.html")
