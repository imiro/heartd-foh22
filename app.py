import flask
import pandas as pd
import math
import numpy as np
import sklearn
import pickle

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
    predictors['aqi'] = np.random.randint(500)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    X = pd.Series(predictors)
    X = X[['smok', 'aqi', 'age', 'male', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
    X = X.to_numpy()
    X = X[None,:]

    y = model.predict(X)

    return str(y[0])

    # race = int(flask.request.form["race"])
    # predictors['race1'] = race == 1
    # predictors['race2'] = race == 2
    # predictors['race3'] = race == 3
    # predictors['famlt'] = int(flask.request.form["famlt"])
    # predictors['emp'] = int(flask.request.form["emp"]) == 1
    # predictors['edu6'] = int(flask.request.form["edu6"])
    # bmi = float(flask.request.form["bmi"])
    # predictors["bmi185"] = bmi <= 18.50
    # predictors["bmilg"] = math.log(bmi)
    # cpd = int(flask.request.form["cpd"])
    # predictors["cpd20"] = cpd > 20
    # smkyears = int(flask.request.form["smkyears"])
    # pkyrcut = smkyears*cpd / 20
    # predictors["smkyears"] = math.log(smkyears)
    # predictors["qtyears1"] = math.log(int(flask.request.form["qtyears"])+1)
    # predictors["age"] = math.log(int(flask.request.form["age"]))
    # predictors["pkyrcut1"] = pkyrcut < 40
    # predictors["pkyrcut2"] = pkyrcut >= 40 and pkyrcut < 50
    # predictors["pkyrcut3"] = pkyrcut >= 50

    # coef = {
    #     'female': -0.178447207,
    #     'race1': 0.393644051,
    #     'race2': -0.375249324,
    #     'race3': -0.420847968,
    #     'edu6': -0.097081521,
    #     'famlt': 0.422066884,
    #     'emp': 0.554449173,
    #     'bmi185': 0.356082765,
    #     'cpd20': 0.241671634,
    #     'pkyrcut1': 0.555515098,
    #     'pkyrcut2': 0.747617036,
    #     'pkyrcut3': 0.894503302,
    #     'age': 6.067989873,
    #     'bmilg': -0.80541569,
    #     'qtyears1': -0.377529292,
    #     'smkyears': 0.332956334
    # }

    # score = 0
    # for (key, value) in coef.items():
    #     score += value * float(predictors[key])
    # 
    # return str(score)
