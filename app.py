from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("accident_severity_model.pkl")  # Ensure this file exists

@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' exists in 'templates/' folder

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    lat = data["latitude"]
    lng = data["longitude"]
    weather = data.get("weather", 4)  # Default: 4 (Overcast)
    road_cond = data.get("road_condition", 0)  # Default: 0 (Dry Road)
    light_cond = data.get("light_condition", 5)  # Default: 5 (Daylight)

    print(f"received input data:{data}")

    # Create input for model prediction
    input_data = [[lng, lat, weather, road_cond,light_cond]]  # Use correct features
    print(f"Input data: {input_data}")  # Log input data

    # Predict accident risk
    prediction = model.predict(input_data)
    print(f"Prediction: {prediction}")  # Log prediction

    return jsonify({"severity": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)