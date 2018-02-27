from flask import Flask, request, abort, jsonify
import json
import os
import pickle
import datetime
import requests
import traceback
import sys
import numpy as np
from model_wrapper_NN import ModelWrapper

app = Flask(__name__)

MODEL_PATH = os.getenv('MODEL_PATH')
URL_STORE_EVENT = "https://icost.ubismart.org/mobility/store"
activities_UCI = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

clf = None

mw = ModelWrapper()

model = mw.load_model(MODEL_PATH)

def send_result(prediction_result):

    epoch = datetime.datetime.utcfromtimestamp(0)
    date = (datetime.datetime.utcnow() - epoch).total_seconds() * 1000.0
    #print("date = ", date)
    print(prediction_result)
    try:
        mobility_type =  activities_UCI[np.argmax(prediction_result)]# choose the most common [0] and the first element of tuple [0]
    except:
        mobility_type = 'Nothing'
    print ("Mobility type = " + mobility_type)
    # test the prediction_result and statistically choose
    data = {"list": [{"house": 89, "date": date, "mobility": mobility_type}]}
    payload = {'data': data}
    r = requests.post(URL_STORE_EVENT, json=payload)
    print(r.url, r)
    return mobility_type


@app.route("/predict_activity", methods=["POST"])
def predict_activity():
    if 'csv_file' in request.files:
        received_file = request.files['csv_file']
        print("File %s received. Reading and predicting..." % received_file.filename)
        try:
            res = list(mw.predict_csv(received_file))
            # print(res)
            toSend = send_result(res)
            return jsonify(toSend)
        except Exception as e:
            print("ERROR. Can't predict using the received CSV.")
            print ('-'*60)
            traceback.print_exc()#file=sys.stdout)
            print ('-'*60)
            print(e)
            abort(400, "Can't predict using the received JSON. Is it a valid JSON?")

    elif 'json_readings' in request.form:
        readings = request.form['json_readings']
        print("JSON received. Reading and predicting...")

        try:
            json_readings = json.loads(readings)
            res = list(mw.predict_json(readings))
            return jsonify(res)
        except Exception:
            print("ERROR. Can't predict using the received JSON.")
            abort(400, "Can't predict using the received JSON. Is it a valid JSON?")

    else:
        abort(401, "No attachment with sensor readings received.")
