from flask import Flask, request, jsonify
import numpy as np
import pickle
import logging
import os

app = Flask(__name__)

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/app.log', level=logging.INFO)

# Load model
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return "✅ Iris Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([list(data.values())]).reshape(1, -1)
    prediction = model.predict(features)[0]
    
    logging.info(f"Prediction made for input: {data} -> {prediction}")
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
