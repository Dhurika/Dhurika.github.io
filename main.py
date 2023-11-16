import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = tf.keras.models.load_model('retinal_oct_model.h5')

# Define your class names based on your problem
CLASS_NAMES = ["CNV", "DME", "DRUSEN","NORMAL"]

app = FastAPI()

@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.post('/predict/')
async def predict(file: UploadFile):
    try:
        logger.info(f"Received file: {file.filename}, content type: {file.content_type}")

        image_data = await file.read()
        logger.info(f"Image data received, length: {len(image_data)} bytes")

        image = Image.open(BytesIO(image_data))
        logger.info(f"Image opened successfully")
        image = image.convert("RGB")
        image = image.resize((224, 224))
        logger.info(f"Image resized to (224, 224)")

        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        logger.info(f"Image processed and normalized")

        predictions = MODEL.predict(image)
        max_prediction = np.max(predictions)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        logger.info(f"Prediction: {predictions}")
        logger.info(f"Max Prediction Value: {max_prediction}")
        logger.info(f"Predicted Class: {predicted_class_name}")

        return {"max_prediction": float(max_prediction), "predicted_class": predicted_class_name}

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail="Error processing image")
