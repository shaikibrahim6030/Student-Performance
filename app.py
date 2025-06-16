from flask import Flask, request
from joblib import load
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
model=load("SalaryModel.pkl")

@app.route('/')
def welcome():
    return "Welcome Everyone"

@app.route('/predict')
def predict_salary():
    experience=request.args.get('experience')
    prediction=model.predict([[float(experience)]])
    prediction =np.round(prediction,2)
    return "The predicted value is" + str(prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
