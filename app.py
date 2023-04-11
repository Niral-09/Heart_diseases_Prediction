from flask import Flask, request, jsonify, render_template
from model import predict_ans

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    output = predict_ans(data)
    if output[0][0] >= 0.5:
        return jsonify({"Ans": "Yes, Heart diseases is detected."})
    else:
        return jsonify({"Ans": "NO, Heart diseases is not detected."})


@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    output = predict_ans(data)
    if output[0] >= 0.5:
        return render_template("false.html")
    else:
        return render_template("true.html")


if __name__ == "__main__":
    app.run(debug=True)
