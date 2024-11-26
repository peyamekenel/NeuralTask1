import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
models = {
    'logistic_regression': joblib.load('models/logistic_regression.pkl'),
    'random_forest': joblib.load('models/random_forest.pkl'),
    'svm': joblib.load('models/svm.pkl')
}
scaler = joblib.load('models/scaler.pkl')

class PredictionInput(BaseModel):
    model_name: str
    glucose: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Validate model name
        if input_data.model_name not in models:
            raise HTTPException(status_code=400, detail="Invalid model name")

        # Prepare input features
        features = np.array([[
            input_data.glucose,
            input_data.bmi,
            input_data.diabetes_pedigree_function,
            input_data.age
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        model = models[input_data.model_name]
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
