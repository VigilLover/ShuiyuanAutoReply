"""Serve locally stored image artifacts (uploads, generated images, cached media)."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ..support import state_store

router = APIRouter()


@router.get("/api/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str, request: Request):
    artifact = await state_store(request).get_artifact(artifact_id)
    if (
        artifact is None
        or not artifact.available
        or not Path(artifact.local_path).is_file()
    ):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(
        artifact.local_path,
        media_type=artifact.mime_type,
        filename=Path(artifact.local_path).name,
        # Artifact ids are content-stable, so images can be cached without
        # revalidation; the monitor re-renders them on every refresh.
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )
