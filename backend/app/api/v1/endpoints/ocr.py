from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.crnn import CRNN
import torch
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class OCRResponse(BaseModel):
    predictions: list
    confidence: float

# Initialize the CRNN model
try:
    model = CRNN(vocab_size=100, hidden_size=256, n_layers=2)
    model.eval()
    # Optionally load pre-trained weights
    # model.load_model("path/to/weights.pth")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise

@router.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)):
    """
    Process an image and return OCR results.
    
    Args:
        file: Image file to process
    Returns:
        OCRResponse with predictions and confidence
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    try:
        # Load and preprocess image
        image = Image.open(BytesIO(await file.read())).convert("L")
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)

        # Perform OCR
        with torch.no_grad():
            predictions = model(image_tensor)
            confidence = torch.mean(torch.max(predictions, dim=2)[0]).item()

        return OCRResponse(
            predictions=predictions.tolist(),
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(500, "Failed to process image")
