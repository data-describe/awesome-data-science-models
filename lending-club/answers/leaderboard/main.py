import argparse
import logging
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import f1_score

app = Flask(__name__)
app.config.from_pyfile('./settings.cfg')


@app.route("/")
def board():
    try:
        if app.config["REVEAL"]:
            df = pd.read_gbq(
                f"SELECT id, f1_public, f1_private FROM `{app.config['PROJECT']}.{app.config['DATASET']}.{app.config['TABLE']}` LIMIT 50"
            )
            df = df.sort_values(by="f1_private", ascending=False)
        else:
            df = pd.read_gbq(
                f"SELECT id, f1_public FROM `{app.config['PROJECT']}.{app.config['DATASET']}.{app.config['TABLE']}` LIMIT 50"
            )
            df = df.sort_values(by="f1_public", ascending=False)

        return render_template(
            "index.html",
            data=df.to_html(
                classes="table table-striped table-hover",
                header="true",
                justify="center",
                index=False,
            ),
        )
    except Exception as e:
        logging.error("Failed to load the leaderboard")
        logging.error(e)
        return render_template("404.html")


@app.route("/submit", methods=["POST"])
def submit():
    _id = request.get_json().get("id")
    scores = request.get_json().get("scores")
    if not scores:
        return (
            "Expected a json with single key 'scores' as an array of predictions",
            400,
        )
    if len(scores) != 2000:
        return "Expected array of size 2000", 400

    labels = pd.read_csv("gs://amazing-public-data/lending_club/exercise/test.csv")[
        "is_bad"
    ].to_list()
    order = list(range(2000))
    np.random.shuffle(order)
    public_idx = order[:1000]
    public = [labels[i] for i in public_idx]

    public_score = f1_score(public, [scores[i] for i in public_idx])
    private_score = f1_score(labels, scores)

    submission = pd.DataFrame(
        {
            "id": [_id],
            "f1_public": [public_score],
            "f1_private": [private_score],
        }
    )
    submission.to_gbq(
        f"{app.config['DATASET']}.{app.config['TABLE']}",
        project_id=app.config["PROJECT"],
        if_exists="append",
    )
    return ""


if __name__ == "__main__":
    app.run(debug=True)
