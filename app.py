from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle

app = FastAPI()

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def get_data(temperature: float = 0.0, humidity: float = 0.0, ph: float = 0.0, rainfall: float = 0.0):
    # Create a DataFrame with the input data
    new_data = pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    return new_data

@app.get('/api')
def predict_crop_api(temperature: float = 0.0, humidity: float = 0.0, ph: float = 0.0, rainfall: float = 0.0):
    try:
        new_data = get_data(temperature, humidity, ph, rainfall)

        # Make predictions for new data using the loaded model
        new_predictions = model.predict(new_data)

        # Return the predictions as a JSON response
        return {'predictions': new_predictions.tolist()}
    except Exception as e:
        # Return an error message if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))
