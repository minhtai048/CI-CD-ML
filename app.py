# APPLICATION FOR MEDICAL COST PREDICTION IN FOUR BIGGEST REGIONS IN USA.

"""
!!!DISCLAIMER!!
All data are protected by US Office Of Government Ethics (OGE).
Commerical purposes are restricted, any modifications are expected to have permission from US Authorities.
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from utils.preprocess import inverse_transform, replace_categories
import pickle

# Create an app object using the Flask class.
app = Flask(__name__)
# Load the trained model and encoder. (Pickle file)
model = pickle.load(open("models/svr_model.pkl", "rb"))
scaler = pickle.load(open("models/mm_encoder.pkl", "rb"))
lmbda_charges = pickle.load(open("models/lmbda_price.pkl", "rb"))


# Define the route to be home.
# Here, home function is with '/', our root directory.
# Running the app sends us to index.html.
# Note that render_template means it looks for the file in the templates folder.
# use the route() decorator to tell Flask what URL should trigger our function.
@app.route("/")
def home():
    return render_template("index.html")


# GET: A GET message is send, and the server returns data
# POST: Used to send HTML form data to the server.
# Add Post method to the decorator to allow for form submission.
# Redirect to /predict page with the output
@app.route("/predict", methods=["POST"])  # Do not add two method may make app crashed
def predict():
    # input section
    ageinput = request.form.get("ageinput")
    genderinput = request.form.get("genderinput")
    bmiinput = request.form.get("bmiinput")
    childinput = request.form.get("childinput")
    smokinginput = request.form.get("smokinginput")
    regioninput = request.form.get("regioninput")
    gender_display = genderinput

    # feature transformation section
    features = pd.DataFrame(
        {
            "age": [ageinput],
            "sex": [genderinput],
            "bmi": [bmiinput],
            "children": [childinput],
            "smoker": [smokinginput],
            "region": [regioninput],
        }
    )
    features = replace_categories(features)
    features = scaler.transform(features)
    prediction = model.predict(features)
    prediction = inverse_transform(prediction, lmbda_charges)
    prediction = np.round_(prediction, 2)[0]

    # final section -> send data back to front page
    return render_template(
        "index.html",
        age=ageinput,
        gender=gender_display,
        bmi=bmiinput,
        child=childinput,
        smoking=smokinginput,
        region=regioninput,
        prediction_text=prediction,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
