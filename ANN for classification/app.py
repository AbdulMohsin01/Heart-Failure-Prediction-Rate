from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the model and scaler
model = load_model('best_ann_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = [
            data['CreditScore'], data['Age'], data['Tenure'], data['Balance'],
            data['NumOfProducts'], data['HasCrCard'], data['IsActiveMember'],
            data['EstimatedSalary'], data['Geography_Germany'], data['Geography_Spain'],
            data['Gender_Male']
        ]
        input_df = pd.DataFrame([input_data], columns=[
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
            'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male'
        ])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        predicted_class = (prediction > 0.5).astype(int)[0][0]
        probability = float(prediction[0][0])
        return jsonify({
            'predicted_class': int(predicted_class),
            'probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)