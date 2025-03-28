import os
import tempfile
from io import BytesIO

import numpy as np
import requests
import torch
from crnn import CRNN
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from ray import serve
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
app = FastAPI()


TEXT_DETECTION_MODEL = 'backend\app\models\weights\best.pt'
OCR_MODEL = 'backend\app\models\weights\ocr_crnn.pt'

CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(sorted(CHARS))}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

# model configuration
HIDDEN_SIZE = 256
N_LAYERS = 3
DROPOUT_PROB = 0.2
UNFREEZE_LAYERS = 3

@serve.deployment(num_replicas=1)
@serve.ingress(app)

class APIIngress:
    def __init__(self, ocr_handler):
        self.ocr_handler = ocr_handler
    
    async def process_image(self, image_data: bytes) -> Response:
        try:
            ###
            with tempfile.NamedTemporaryFile(delete=False) as temp_image:
                temp_image.write(image_data)
                temp_image_path = temp_image.name

            prediction = await self.ocr_handler.process_image.remote(temp_image_path)

            image = Image.open(temp_image_path)
            annotated_image = await self.ocr_handler.draw_predictions.remote(
                image, prediction
            )
        
            file_stream = BytesIO()
            annotated_image.save(file_stream, format="PNG")
            file_stream.seek(0)


            os.unlink(temp_image_path)

            return Response(
                content=file_stream.getvalue(),
                media_type="image/png",
                headers={"X-Predictions": str(prediction)},
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}"))
        

    @app.get("/ocr")
    async def ocr_url(self, image_url: str):
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return await self.process_image(response.content)
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"{e}")
    
    @app.post("/ocr/upload")
    async def ocr_upload(self, file: UploadFile = File(...)):
        ###
        return await self.process_image(await file.read())



@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)

class OCRHandler:
    def __init__(self, reg_model, det_model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.det_model = det_model.to(self.device)
        self.reg_model = reg_model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((100, 420)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )    

    def text_detection(self, image_path):
        results = self.det_model(image_path, verbose=False)[0]
        return(
            results.boxes.xyxy.tolist(),
            results.boxes.cls.tolist(),
            results.names,
            results.boxes.conf.tolist(),
        )

    def text_recognition(self, img):
        transformed_img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.reg_model(transformed_img).cpu()
        text = self.decode_prediction(prediction.permute(1, 0, 2).argmax(2), IDX_TO_CHAR)
        return text
    



    def process_image(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Perform text detection
        det_results = self.det_model(image)

        # Perform OCR
        prediction = self.reg_model(det_results)

        return prediction
    












