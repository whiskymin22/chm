import os
import tempfile
from io import BytesIO

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from ray import serve
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors





from fastapi import FastAPI
from app.api.v1.endpoints import router as api_router

app = FastAPI()

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the CHM API!"}   