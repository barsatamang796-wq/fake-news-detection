from flask import Flask, render_template, request, jsonify
from model import predict_news

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]

    result = predict_news(text)

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)