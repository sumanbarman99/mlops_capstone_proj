import joblib
import numpy as np

# Define the feature columns used in the model
feature_columns = [
    "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin"
]

def predict_mpg(input_data):
    """
    Predict mpg using the loaded model and input data.

    input_data: dict
        Dictionary with keys matching feature_columns and numeric values.

    Returns:
        Predicted mpg value (float).
    """
    # Load the trained model
    model = joblib.load('bestAutoMPGModel.pkl')

    # Convert input dict to DataFrame with one row
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Make prediction
    mpg_pred = model.predict(features)

    return mpg_pred[0]

if __name__ == "__main__":
    # Example input data - replace with real values or take input dynamically
    example_input = {
        "cylinders": 4,
        "displacement": 150.0,
        "horsepower": 90.0,
        "weight": 2800,
        "acceleration": 15.5,
        "model_year": 82,
        "origin": 1
    }

    predicted_mpg = predict_mpg(example_input)
    print(f"Predicted MPG: {predicted_mpg:.2f}")
