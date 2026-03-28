import joblib
import numpy as np
from fastapi import FastAPI

app = FastAPI()

# Load trained model
model = joblib.load("models/model.pkl")


@app.get("/")
def home():
    return {"message": "ML API Running "}


@app.get("/predict")
def predict(calories: float, workout: float, workout_freq: float, bmi: float):
    data = np.array([[calories, workout, workout_freq, bmi]])
    prediction = model.predict(data)

    return {"predicted_body_fat": float(prediction[0])}