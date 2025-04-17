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

app = FastAPI()

@serve.deployment(num_replicas=1)
@serve.ingress(app)

class APIIngress:
    def __init__(self, object_detection_handler):
        self.object_detection_handler = object_detection_handler

    async def process_image(self, image_data: bytes) -> Response:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(image_data)
                temp_file_path = temp_file.name
            
            bboxes, classes, names, confs = await self.handle.detect.remote(temp_file_path)
            image = Image.open(temp_file_path)
            image_array = np.array(image)

            annotator = Annotator(image_array, font="Arial.ttf", pil=True)

            for bbox, cls, conf in zip(bboxes, classes, confs):
                x1, y1, x2, y2 = bbox
                color = colors(cls, True)
                annotator.box_label([x1, y1, x2, y2], names[cls], color=color)

            # Covnert the image back to bytes
            annotated_image = Image.fromarray(annotator.result())
            file_stream = BytesIO()
            annotated_image.save(file_stream, format="PNG")
            file_stream.seek(0)

            # Clean up the temporary file
            os.unlink(temp_file_path)
            return Response(content=file_stream.getvalue(), media_type="image/png")
        
        except Exception as e:
            raise HTTPException(status_code=500, detail= f"Error processing image: {str(e)}")
        
    
    @app.get("/detect", response_class=Response)
    async def detect_url(self, url:str):
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Invalid image URL")
            
            image_data = response.content
            return await self.process_image(image_data)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching image from URL: {str(e)}")
        
    @app.post("/detect/upload", response_class=Response)
    async def detect_upload(self, file: UploadFile = File(...)):
        try:
            image_data = await file.read()
            return await self.process_image(image_data)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing uploaded image: {str(e)}")
        


@serve.deployment(
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.5},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 2,
    },
)

class ObjectDetectionHandler:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
    
    def detect(self, image_path: str):
        results = self.model(image_path, conf=0.25)
        bboxes = []
        classes = []
        names = results[0].names
        confs = []
        try:
            for result in results:
                for box in result.boxes:
                    bboxes.append(box.xyxy[0].cpu().numpy())
                    classes.append(int(box.cls[0].cpu().numpy()))
                    confs.append(float(box.conf[0].cpu().numpy()))

            return bboxes, classes, names, confs
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing detection results: {str(e)}")

entrypoint = ObjectDetectionHandler.bind()
object_detection_handler = APIIngress.bind(entrypoint)
