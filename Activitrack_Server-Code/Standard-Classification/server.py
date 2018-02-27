from flask import Flask, request, abort, jsonify
import json
import os
import pickle
import datetime
import requests
import traceback
import sys
from model_wrapper_win import ModelWrapper

app = Flask(__name__)

MODEL_PATH = os.getenv('MODEL_PATH')
URL_STORE_EVENT = "https://icost.ubismart.org/mobility/store"

clf = None

mw = ModelWrapper()

features, clf = mw.load_model(MODEL_PATH)

@app.route("/")
def hello():
    return "Hello World!"

# wget -q
# "https://icost.ubismart.org/mobility/store?data={\"list\":[{\"house\":89,\"date\":$(date
# +%s)000,\"mobility\":\"bicycle\"},{\"house\":89,\"date\":$(( $(date +%s)700 - 20000
# )),\"mobility\":\"car\"}]}"
def send_result(prediction_result):
   # return
    #print(prediction_result)
    active_mobility = ['Bike', 'Walk']
    epoch = datetime.datetime.utcfromtimestamp(0)
    date = (datetime.datetime.utcnow() - epoch).total_seconds() * 1000.0
    print("date = ", date)
    try:
        from collections import Counter
        print (Counter(prediction_result))
        counts = Counter(prediction_result).most_common()
        print (counts)
        mobility_type = counts[0][0] # choose the most common [0] and the first element of tuple [0]
    except:
        mobility_type = 'Nothing'
    print ("Mobility type = Active" if str(mobility_type) in active_mobility else "Mobility type = Pasive")
    # test the prediction_result and statistically choose
    data = {"list": [{"house": 89, "date": date, "mobility": mobility_type}]}
    payload = {'data': data}
    r = requests.post(URL_STORE_EVENT, json=payload)
    print(r.url, r)


@app.route("/predict_activity", methods=["POST"])
def predict_activity():
    if 'csv_file' in request.files:
        received_file = request.files['csv_file']
        print("File %s received. Reading and predicting..." % received_file.filename)
        try:
            res = list(mw.predict_csv(received_file))
            # print(res)
            send_result(res)
            return jsonify(res)
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
