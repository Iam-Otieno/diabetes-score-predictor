# Diabetes Prediction API using FastAPI
#impoting necessary libraries
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import os
import joblib
import numpy as np


# Define input data schema using Pydantic
class PatientData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float # Blood Pressure
    s1: float  # Serum measurement 1
    s2: float  # Serum measurement 2
    s3: float  # Serum measurement 3
    s4: float  # Serum measurement 4
    s5: float  # Serum measurement 5
    s6: float  # Serum measurement 6
    class Config:
        json_schema_extra = {
            "example": {
                "age": 0.05,
                "sex": 0.05,
                "bmi": 0.05,
                "bp": 0.02,
                "si": 0.04,
                "s2": 0.04,
                "s3": 0.02,
                "s4": 0.01,
                "s5": 0.01,
                "s6": 0.02
            }
        }
# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts diabetes progression based on patient data",
    version="1.0.0"
)

# Load the trained model
model_dir = 'models'
model_path = os.path.join(model_dir, 'diabetes_model.joblib')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
with open(model_path, 'rb') as f:
    model = joblib.load(f)
    
# Define the prediction endpoint
@app.post("/predict", response_model=float)
def predict(patient_data: PatientData):
    """
    Predict diabetes progression based on patient data.
    
    Args:
        patient_data (PatientData): Input data for prediction.
        
    Returns:
        float: Predicted diabetes progression value.
    """
    # Convert input data to numpy array
    input_data = np.array([[patient_data.age, 
                            patient_data.sex,
                            patient_data.bmi, 
                            patient_data.bp,
                            patient_data.s1,
                            patient_data.s2,
                            patient_data.s3,
                            patient_data.s4,
                            patient_data.s5,
                            patient_data.s6]])
    # Ensure input data is in the correct shape
    if input_data.shape[1] != 10:
        raise ValueError("Input data must have exactly 10 features.")
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)[0]
    
    return {
        "Predicted_progressoion_score": round(prediction, 2),
        "iInterpretation": get_interpretation(prediction)
    }
def get_interpretation(score: float) -> str:
    """
    Get human-readable interpretation of the prediction score.
    Args:
        score (float): Predicted diabetes progression score.
    Returns:
        str: Interpretation of the score.
    """
    if score < 100:
        return "Normal progression"
    elif 100 <= score < 200:
        return "Mild progression"
    elif 200 <= score < 300:
        return "Moderate progression"
    else:
        return "Severe progression"
    
    @app.get("/")
    def health_check():
        """
        Health check endpoint to verify the API is running.
        Returns:
            str: Health status message.
        """
        return {"status": "API is running", "model_loaded": True}
    
# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="
    
                
