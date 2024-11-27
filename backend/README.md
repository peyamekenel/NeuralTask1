# Diabetes Prediction API

This FastAPI backend provides endpoints for making diabetes predictions using multiple machine learning models including:
- Logistic Regression
- Random Forest
- SVM
- Keras Neural Network

## API Endpoints

### POST /predict
Makes a prediction using the specified model.

Request body:
```json
{
    "model_name": "string",  // One of: "logistic_regression", "random_forest", "svm", "keras"
    "glucose": float,
    "bmi": float,
    "dpf": float,
    "age": float
}
```

Response:
```json
{
    "prediction": int,      // 0 or 1
    "probability": float    // Probability of positive class
}
```

### GET /health
Health check endpoint.

Response:
```json
{
    "status": "healthy"
}
```
