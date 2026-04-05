from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI()

# Automatically find model file
model = None

def load_model():
    global model
    if model is None:
        paths = [
            "model/model.pkl",
            "models/model.pkl"
        ]
        
        for path in paths:
            if os.path.exists(path):
                model = joblib.load(path)
                print(f"Model loaded from {path}")
                return model
        
        print("Model file not found, using dummy mode")
        return None
    
    return model


#  Home API
@app.get("/")
def home():
    return {"message": "ML API is running"}


#  Prediction API
@app.post("/predict")
def predict(data: dict):
    mdl = load_model()
    
    if mdl is None:
        return {"prediction": [42]}
    
    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = mdl.predict(features)
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        return {"error": str(e)}


#  Body Fat API
@app.post("/bodyfat")
def body_fat(data: dict):
    weight = data["weight"]
    height = data["height"] / 100
    age = data["age"]

    bmi = weight / (height ** 2)
    body_fat = (1.20 * bmi) + (0.23 * age) - 16.2

    return {
        "BMI": round(bmi, 2),
        "BodyFat%": round(body_fat, 2)
    }


#  Calories Burn API
@app.post("/calories")
def calories(data: dict):
    weight = data["weight"]
    duration = data["duration"]
    heart_rate = data["heart_rate"]

    calories = (0.630 * heart_rate + 0.198 * weight + 0.201 * duration - 55)

    return {
        "calories_burned": round(calories, 2)
    }


#  Workout Recommendation API
@app.post("/workout")
def workout(data: dict):
    goal = data["goal"]

    if goal == "weight_loss":
        return {"workout": "Cardio + HIIT"}
    elif goal == "muscle_gain":
        return {"workout": "Strength Training"}
    else:
        return {"workout": "General Fitness Routine"}


#  BMI Category API
@app.post("/bmi-category")
def bmi_category(data: dict):
    weight = data["weight"]
    height = data["height"] / 100

    bmi = weight / (height ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return {
        "BMI": round(bmi, 2),
        "category": category
    }


#  Health Risk API
@app.post("/health-risk")
def health_risk(data: dict):
    age = data["age"]
    bmi = data["bmi"]

    if bmi > 30 or age > 50:
        risk = "High"
    elif bmi > 25:
        risk = "Medium"
    else:
        risk = "Low"

    return {"risk_level": risk}