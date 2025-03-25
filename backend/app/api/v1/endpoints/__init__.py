# This file is intentionally left blank.

from fastapi import APIRouter
from .default import router as default_router  # Import your new file
from .ocr import router as ocr_router  # Import the OCR router

router = APIRouter()
router.include_router(default_router, prefix="/default", tags=["Default"])
router.include_router(ocr_router, prefix="/ocr", tags=["OCR"])
