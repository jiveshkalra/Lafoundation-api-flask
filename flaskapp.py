from flask import Flask, request, jsonify, render_template
import pandas as pd
# import pickle
import joblib

app = Flask(__name__)

# Load the model from the pickle file
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)
model = joblib.load('final_model.joblib')

def get_data():
    # Get the input parameters from the GET request
    temperature = float(request.args.get('temperature', default=0.0))
    humidity = float(request.args.get('humidity', default=0.0))
    ph = float(request.args.get('ph', default=0.0))
    rainfall = float(request.args.get('rainfall', default=0.0))
    return temperature, humidity, ph, rainfall

@app.route('/api')
def predict_crop_api():
    temperature, humidity, ph, rainfall = get_data()
    try:
        # Make predictions for new data using the loaded model
        new_predictions = model.predict([[temperature, humidity, ph, rainfall]])

        # Return the predictions as a JSON response
        return jsonify({'predictions': new_predictions.tolist()})
    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({'error': str(e)})

@app.route('/')
def predict_crop():
    temperature, humidity, ph, rainfall = get_data()
    try:
        # Make predictions for new data using the loaded model
        new_predictions = model.predict([[temperature, humidity, ph, rainfall]])

        # Return the predictions as a JSON response
        return render_template('index.html', temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall, predictions=new_predictions)
    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)