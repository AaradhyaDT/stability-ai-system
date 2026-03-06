from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model/stability_model.joblib")

class SensorData(BaseModel):
    tilt_x: float
    tilt_y: float
    angular_velocity: float

@app.get("/")
def home():
    return {"message": "Stability AI API running"}

@app.post("/predict")
def predict(data: SensorData):

    features = [[data.tilt_x, data.tilt_y, data.angular_velocity]]

    prediction = model.predict(features)[0]

    result = "stable" if prediction == 1 else "unstable"

    return {"prediction": result}