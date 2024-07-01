import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Buat aplikasi Flask
app = Flask(__name__)

# Muat model yang di-pickle
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index2.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Dapatkan fitur input dari formulir dan konversikan ke float
    float_features = [float(x) for x in request.form.values() if x]
    if len(float_features) != 5:
        return render_template("index2.html", prediction_result="Unknown")

    features = [np.array(float_features)]

    # Lakukan prediksi menggunakan model yang dimuat
    prediction = model.predict(features)

    # Teruskan hasil prediksi ke template HTML
    return render_template("index2.html", prediction_result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
