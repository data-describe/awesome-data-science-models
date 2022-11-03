import joblib
import numpy as np
import pickle
from flask import Flask, jsonify, request


app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/predict", methods=["POST", "GET"])
def predict():
    data = request.get_json(force=True)["instances"]

    prediction = model.predict([data]).tolist()[0]
    print("prediction:", prediction)

    # postprocessing
    output = {"predictions": [prediction]}
    return jsonify(output)


@app.route("/healthz")
def healthz():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0")
