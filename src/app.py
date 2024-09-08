# app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__, template_folder='templates')

# Global variable to hold our model
model = None

def load_model():
    global model
    if model is None:
        data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv")
        X = data[['age', 'bmi', 'children', 'smoker', 'region']]
        y = data['charges']
        X = pd.get_dummies(X, drop_first=True)

        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, 'insurance_cost_model.joblib')

@app.before_request
def before_request():
    load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json
    df = pd.DataFrame(features, index=[0])
    df = pd.get_dummies(df, drop_first=True)
    
    # Ensure all columns from training are present
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    
    prediction = model.predict(df[model.feature_names_in_])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)