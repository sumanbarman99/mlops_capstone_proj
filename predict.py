import joblib
import pandas as pd

feature_columns = [
    "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin"
]

def predict_mpg(input_data):
    model = joblib.load('bestAutoMPGModel.pkl')

    # Convert input dict to DataFrame with columns
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    print(input_df)
    # Pass the DataFrame (not a numpy array) to model.predict()
    mpg_pred = model.predict(input_df)

    return mpg_pred[0]

if __name__ == "__main__":
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
