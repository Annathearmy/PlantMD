from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os

app = FastAPI(
    title="PlantMD Disease Detection API",
    version="1.0.0"
)

MODEL_PATH = "models/AlexNetModel.hdf5"

# These labels match the original PlantMD prediction code.
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

model = None


@app.on_event("startup")
def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}"
        )

    print("Loading PlantMD disease detection model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("PlantMD model loaded successfully.")


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PlantMD Disease Detection API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file"
        )

    try:
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Same preprocessing used by the original PlantMD code.
        image = image.resize(
            (224, 224),
            Image.NEAREST
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = image_array / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        prediction = model.predict(
            image_array,
            verbose=0
        )

        probabilities = prediction[0]

        class_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[class_index]
        )

        if class_index >= len(CLASS_NAMES):
            raise HTTPException(
                status_code=500,
                detail="Model output does not match class list"
            )

        raw_prediction = CLASS_NAMES[class_index]

        # Convert:
        # Tomato___Late_blight
        # → Tomato - Late blight
        display_prediction = (
            raw_prediction
            .replace("___", " - ")
            .replace("_", " ")
        )

        crop = raw_prediction.split("___")[0]

        healthy = "healthy" in raw_prediction.lower()

        if healthy:
            disease_status = "healthy"
        else:
            disease_status = "disease_detected"

        return {
            "success": True,
            "prediction": display_prediction,
            "crop": crop,
            "disease": None if healthy else display_prediction,
            "status": disease_status,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
