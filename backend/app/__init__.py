# backend/app/__init__.py

from fastapi import FastAPI

app = FastAPI()

from .api.v1.endpoints import router as api_router

app.include_router(api_router, prefix="/api/v1")