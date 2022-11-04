import joblib
import numpy as np
import pickle
from flask import Flask, jsonify, request


app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/predict", methods=["POST", "GET"])
def predict():
    data = request.get_json(force=True)["instances"]
    pred = model.predict([data]).tolist()[0]
    product_1_cat = pred.index(max(pred)) + 1
    output = {"predictions": [f"Product Category {product_1_cat}"]}
    return jsonify(output)


@app.route("/healthz")
def healthz():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0")
