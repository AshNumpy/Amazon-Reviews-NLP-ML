from flask import Flask, request, jsonify, render_template 
import pandas as pd
import numpy as np
import pickle
from cleaning import clean_text
from sklearn.feature_extraction.text import CountVectorizer

# Creating the app 
app = Flask(__name__)

# Load the pickle model
model =  pickle.load(open("model.pkl", "rb"))

# Define the home page
@app.route("/")
def Homepage():
    return render_template("index.html")

# Define Prediction page
@app.route("/prediction", methods=['POST'])
def prediction():
    review = [review for review in request.form.values()]
    review = {'Review':review}
    df = pd.DataFrame(review)

    df['Review'] = clean_text(df['Review'])
    X = df.loc[:,'Review']

    cv = CountVectorizer(min_df=0., max_df=1.)
    X_vector = cv.fit_transform(X.values.astype('U'))

    prediction = model.predict(X_vector)
    return render_template('index.html', prediction_text=f'Prediction is {prediction} Star {"*"*prediction}')

if __name__ == "__main__":
    app.run(debug=True)