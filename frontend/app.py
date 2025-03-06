import streamlit as st
import pandas as pd
import numpy as np

from io import BytesIO
import requests
from PIL import Image

st.set_page_config(layout="wide")

def format_predictions(predictions_str):
    """Format the predictions string into a list of dictionaries"""
    predictions = []
    ## cach 1
    # for line in predictions_str.split("\n"):
    #     if line.strip():
    #         predictions.append(json.loads(line))
    
    
    ## cach 2
    try: 
        predictions = eval(predictions_str)

        if not predictions:
            return "[]"
        
        formatted_json = "[\n"
        for bbox, class_name, confidence, text in predictions:
            formatted_json += f"    {{'bbox': {bbox}, 'class_name': '{class_name}', 'confidence': {confidence}, 'text': '{text}'}},\n"
        formatted_json += "]"
        formatted_json = formatted_json.rstrip(",\n") + "\n]"
        return formatted_json

    except Exception as e:
        return f"Error formatting predictions: {str(e)}" 


    def process_image_url(url, api_url="http://localhost:8000"):
        """Process image from URL using the OCR API"""
        try:
            response = requests.get(f"{api_url}/ocr", params={"image_url":url})
            response.raise_for_status()

            #get predictions from headers
            predictions = response.headers.get("X-Predictions","[]")

            # Display the processed image
            image = Image.open(BytesIO(response.content))
            
            return image, predictions
        except requests.RequestException as e:
            st.error(f"Error processing image : {str(e)}")
            return None, None
        
    
    def process_upload_file(file, api_url="http://localhost:8000"):
        




st.title("Welcome to the CHM Frontend")

st.write("This is a simple Streamlit application.")

data = np.random.randn(10, 2)
df = pd.DataFrame(data, columns=["Column 1", "Column 2"])

st.line_chart(df)
