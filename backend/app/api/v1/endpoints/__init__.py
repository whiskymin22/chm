# This file is intentionally left blank.

from fastapi import APIRouter
from .default import router as default_router  # Import your new file

router = APIRouter()
router.include_router(default_router, prefix="/default", tags=["Default"])
