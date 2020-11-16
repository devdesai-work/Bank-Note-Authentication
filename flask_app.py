# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:03:43 2020

@author: jj
"""

import numpy as np
import pandas as pd
import pickle
from flask import Flask,request

app = Flask(__name__)
pickle_in = open('note_classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome!!"

@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is" + str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    test_file = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(test_file) 
    return "The predicted value for the test csv file is" + str(list(prediction))

if __name__ == '__main__':
    app.run()
    