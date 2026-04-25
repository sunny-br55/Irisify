
from flask import Flask, render_template, request, jsonify
import requests
import pickle
import numpy as np

API_KEY = "2b10lAksdn5bpR1BuSoZ0WSQl"

app = Flask(__name__)

# LOAD MODEL FILES
model = pickle.load(open("models/best_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array(data["features"]).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)
    result = le.inverse_transform(prediction)[0]

    # ✅ NORMALIZE RESULT (IMPORTANT FIX)
    if "setosa" in result.lower():
        result = "Setosa"
    elif "versicolor" in result.lower():
        result = "Versicolor"
    elif "virginica" in result.lower():
        result = "Virginica"

    # confidence
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(features)) * 100
    else:
        confidence = 90

    return jsonify({
        "result": result,
        "confidence": f"{confidence:.2f}%"
    })


@app.route("/predict-image", methods=["POST"])
def predict_image():
    file = request.files["image"]

    url = "https://my-api.plantnet.org/v2/identify/all"

    files = {
        "images": (file.filename, file.stream, file.mimetype)
    }

    params = {
        "api-key": API_KEY
    }

    response = requests.post(url, files=files, params=params)

    if response.status_code != 200:
        return jsonify({
            "result": "Unknown",
            "confidence": "0%"
        })

    data = response.json()

    try:
        result_name = data["results"][0]["species"]["scientificName"]
        score = data["results"][0]["score"]

        if "setosa" in result_name.lower():
            final = "Setosa"
        elif "versicolor" in result_name.lower():
            final = "Versicolor"
        elif "virginica" in result_name.lower():
            final = "Virginica"
        else:
            final = "Unknown"

        confidence = round(score * 100, 2)

    except:
        final = "Unknown"
        confidence = 0

    return jsonify({
        "result": final,
        "confidence": str(confidence) + "%"
    })


if __name__ == "__main__":
    app.run(debug=True)