from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Age": int(request.form["Age"]),
        "Gender": request.form["Gender"],
        "Tenure": int(request.form["Tenure"]),
        "Usage Frequency": int(request.form["Usage Frequency"]),
        "Support Calls": int(request.form["Support Calls"]),
        "Payment Delay": int(request.form["Payment Delay"]),
        "Subscription Type": request.form["Subscription Type"],
        "Contract Length": request.form["Contract Length"],
        "Total Spend": float(request.form["Total Spend"]),
        "Last Interaction": int(request.form["Last Interaction"])
    }

    predictor = PredictPipeline()
    prediction = predictor.predict(data)

    result = "Churn" if prediction == 1 else "No Churn"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
