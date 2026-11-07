import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.services import imaging
from research.trust.nikhitha_trust_panel.service import assess_image

router = APIRouter()


@router.post("/api/trust/assess")
async def assess(file: UploadFile = File(...)):
    data = await file.read(settings.max_upload_mb * 2**20 + 1)

    if len(data) > settings.max_upload_mb * 2**20:
        raise HTTPException(413, "File too large")

    try:
        img, _ = imaging.load(data, file.filename or "")
        issues = imaging.validate(img)

        if issues:
            raise ValueError("; ".join(issues))

    except ValueError as e:
        raise HTTPException(422, str(e))

    return assess_image(
        img,
        stub=os.getenv("LUCID_TRUST_STUB") == "1",
    )