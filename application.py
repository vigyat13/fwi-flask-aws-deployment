from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

appliction = Flask(__name__)
app=appliction

# Load model and scaler
model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaling.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Get the inputs from the form and convert to float
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))
            DC = float(request.form.get('DC'))
            BUI = float(request.form.get('BUI'))
 

            # Create input array
            input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region,DC,BUI]])

            # Scale input data
            scaled_data =scaler.transform(input_data)

            # Predict FWI
            prediction = model.predict(scaled_data)

            predicted_fwi = prediction[0]

            return render_template('index.html', result=f'Predicted FWI: {predicted_fwi:.2f}')
        
        except Exception as e:
            return render_template('index.html', result=f'Error: {str(e)}')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
