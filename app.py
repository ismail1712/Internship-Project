from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import math
from flask import flash

app = Flask(__name__)
app.secret_key = "Ismail"
model = pickle.load(open('auto\model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    enginesize = request.form['enginesize']
    wheelbase = request.form['wheelbase']
    peakrpm = request.form['peakrpm']
    horsepower = request.form['horsepower']
    citympg = request.form['citympg']
    numdoors = request.form['numdoors']

    if enginesize == "" or wheelbase == ""or peakrpm == ""or horsepower == ""or citympg == ""or numdoors == "":
                flash("Invalid: Every field is required.")
                return redirect(url_for("index"))
    
    arr = np.array([enginesize,wheelbase,peakrpm,horsepower,citympg,numdoors])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', prediction=int(pred))

if __name__ == '__main__':
    app.run(debug=True)