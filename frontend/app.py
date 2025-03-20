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
        # Raises an HTTPError if the HTTP request returned an unsuccessful status code
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
    """ Process uploaded image file using the OCR API"""
    try:
        try:
            Image.open(file)

            file.seek(0)
        except Exception as e:
            st.error("Please upload a valid image file")

        # Prepare the file for upload
        files = {"file": ("image.png", file, "image/png")}
        
        response = requests.post(f"{api_url}/ocr/upload", files = files)

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"Server Error: {error_detail}")
            return None, None
        
        response.raise_for_status()

        #Get predictions from headers
        predictions = response.headers.get("X-Predictions", "[]")

        # Display the processed images
        image = Image.open(BytesIO(response.content))
        return image, predictions

    except requests.RequestException as e:
        st.error(f"Error processing image: {str(e)}")
        if hasattr(e.response, "josn"):
            try:
                error_detail = e.response.json().get("detail", "")
                st.error(f"Server details : {error_detail}")

            except:
                pass

        return None, None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None, None

def main():
    st.title("OCR Image Processing")

    # API endpoint configuration
    api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

    # Two tabs for URL and file upload
    tab1 , tab2 = st.tabs(["Process URL", "Upload Image"])
    
    
    with tab1:
        st.header("Process Image from URL")
        image_url = st.text_input("Enter Image URL")

        #Create 2 columns for images
        col1, col2 = st.columns(2)

        
        with col1:
            st.subheader("Original Image")
            if image_url:
                try:
                    response = requests.get(image_url)
                    original_image = Image.open(BytesIO(response.content))
                    st.image(original_image, use_container_width=True)
                
                except:
                    st.error("Could not load image from URL")

        with col2:
            st.subheader("Processed Image")
            st.empty()

        if image_url and st.button ("Process URL", key="process_url"):
            with st.spinner("Processing image..."):
                image, predictions = process_image_url(image_url, api_url)
                if image:
                    with col2:
                        st.image(image, use_container_width=True)
                    
                    st.subheader("Detected Text")
                    st.code(format_predictions(predictions, language="text"))
    
    with tab2:
        st.header("Upload Image")

        upload_col, button_col = st.columns([4,1])

        with upload_col:
            uploaded_file = st.file_uploader(
                "Choose an image file", type = ["jpeg", "jpg", "png"]
            )
        
        with button_col:
            st.write("...")
            st.write()
            process_button = st.button("Process Image", key = "process_upload")
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(uploaded_file, use_container_width=True)

            with col2:
                st.subheader("Processed image")
                st.empty()
            
            if process_button:
                with st.spinner("Processing image..."):
                    image, predictions = process_upload_file(uploaded_file, api_url)
                    if image:
                        with col2:
                            st.image(image, use_container_width=True)
                        
                        st.subheader("Detected Text")
                        st.code(format_predictions(predictions), language="text")
                    
        










if __name__ == "__main__":
    main()