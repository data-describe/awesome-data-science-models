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
    product_1_cat = pred.index(max(pred)) + 1

    # postprocessing
    output = {"predictions": [f'Product Category {product_1_cat}']}
    print("prediction", output)
    return jsonify(output)



    outputs = self._model.predict(instances)
    pred = outputs.tolist()[0]
    product_1_cat = pred.index(max(pred)) + 1
    return 'Product Category {}'.format(str(product_1_cat))


@app.route("/healthz")
def healthz():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0")
