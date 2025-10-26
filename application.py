import pickle
from flask import Flask, request, url_for, render_template, redirect
import numpy as np

app = Flask(__name__)

# Load model + scaler (make sure models/ridge.pkl and models/scaler.pkl exist)
try:
    ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
    scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))
except Exception as e:
    # If loading fails, print error and set to None so server still starts
    print("Model loading error:", e)
    ridge_model = None
    scaler_model = None

@app.route('/')
def index():
    # attributes/greeting page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoints():
    # Prediction page with form and results
    if request.method == 'POST':
        try:
            # convert inputs to floats
            Temperature = float(request.form.get('Temperature', 0))
            RH = float(request.form.get('RH', 0))
            Ws = float(request.form.get('Ws', 0))
            Rain = float(request.form.get('Rain', 0))
            FFMC = float(request.form.get('FFMC', 0))
            DMC = float(request.form.get('DMC', 0))
            ISI = float(request.form.get('ISI', 0))
            Classes = float(request.form.get('Classes', 0))
            Region = float(request.form.get('Region', 0))

            # ensure models are loaded
            if scaler_model is None or ridge_model is None:
                return render_template('home.html', results="Model not loaded. Check server logs.")

            # scale and predict
            new_data_scaled = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data_scaled)

            # return numeric result
            return render_template('home.html', results=round(float(result[0]), 4))
        except Exception as err:
            # show error message on page to help debug
            return render_template('home.html', results=f"Error: {err}")
    else:
        return render_template('home.html', results=None)

if __name__ == '__main__':
    # Use debug=True while developing; change in production
    app.run(host='0.0.0.0', port=5000, debug=True)
