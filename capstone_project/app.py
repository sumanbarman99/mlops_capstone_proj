from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import numpy as np

# # Load model from MLflow Model Registry (Production stage)
# model_name = "BestAutoMPGModel"
# model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
import joblib

model = joblib.load("bestAutoMPGModel.pkl")


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Expecting one or more records as a list of dicts
        input_df = pd.DataFrame(data)
        
        # Predict MPG
        predictions = model.predict(input_df)
        
        return jsonify({"mpg_predictions": predictions.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def index():
    return "AutoMPG Model API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
