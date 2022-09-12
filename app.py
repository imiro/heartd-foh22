import flask
import pandas as pd
import math
import numpy as np
import sklearn
import pickle
import requests

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return flask.render_template('main.html')

@app.route('/predict', methods=["POST"])
def predict():
    predictors = {}
    labels_int = ['age', 'smok', 'cp', 'trestbps', 'restecg', 'thalach', 'exang',
                'slope', 'ca', 'thal']
    labels_float = ['chol', 'oldpeak']
    predictors['male'] = int(int(flask.request.form["gender"]) == 0)
    predictors['fbs'] = int(int(flask.request.form["fbs"]) > 120)
    for lbl in labels_int:
        predictors[lbl] = int(flask.request.form[lbl])
    for lbl in labels_float:
        predictors[lbl] = float(flask.request.form[lbl])
    predictors['aqi'] = get_aqi_from_zipcode(flask.request.form["zipcode"])

    if( flask.request.form["famh"] == "1" ):
        return "1"

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    X = pd.Series(predictors)
    X = X[['smok', 'aqi', 'age', 'male', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
    X = X.to_numpy()
    X = X[None,:]

    y = model.predict(X)

    return str(y[0])

def get_aqi_from_zipcode(zipcode):
    r = requests.get(f"https://www.airnowapi.org/aq/forecast/zipCode/?format=application/json&zipCode={zipcode}&date=2022-09-11&distance=5&API_KEY=A5AB478C-F11F-4999-8B6C-02F526871F31")
    if not(r.ok):
        return -1
    data = r.json()
    if len(data) < 1:
        return -1
    return data[0]['AQI']
