import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import logging
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scaler
models = {}
try:
    models = {
        'logistic_regression': joblib.load('models/logistic_regression.pkl'),
        'random_forest': joblib.load('models/random_forest.pkl'),
        'svm': joblib.load('models/svm.pkl')
    }

    # Load Keras model if it exists
    keras_model_path = 'models/keras_model.keras'
    if os.path.exists(keras_model_path):
        models['keras'] = tf.keras.models.load_model(keras_model_path)
        logger.info("Keras model loaded successfully")
    else:
        logger.warning("Keras model file not found")

    scaler = joblib.load('models/scaler.pkl')
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

class PredictionInput(BaseModel):
    model_name: str
    glucose: float
    bmi: float
    dpf: float
    age: float

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        logger.info(f"Received prediction request for model: {input_data.model_name}")

        # Prepare features
        features = np.array([[input_data.glucose, input_data.bmi, input_data.dpf, input_data.age]])
        features_scaled = scaler.transform(features)

        model = models.get(input_data.model_name)
        if not model:
            raise HTTPException(status_code=400, detail=f"Model {input_data.model_name} not found")

        try:
            if input_data.model_name == 'keras':
                logger.info("Making prediction with Keras model")
                prediction_prob = float(model.predict(features_scaled, verbose=0)[0][0])
                prediction = 1 if prediction_prob >= 0.5 else 0
                probability = prediction_prob
            else:
                prediction = int(model.predict(features_scaled)[0])
                probability = float(model.predict_proba(features_scaled)[0][1])

            logger.info(f"Prediction successful: {prediction}, probability: {probability}")
            return {"prediction": prediction, "probability": probability}

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail="Error making prediction")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
