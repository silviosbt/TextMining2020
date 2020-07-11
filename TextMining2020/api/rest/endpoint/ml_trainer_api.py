'''
Created on 11 lug 2020

@author: Utente
'''

import flask
from flask import Blueprint, request
import json, time, uuid
import main
from config.websetting import ML_TRAIN_QUEUE, CLIENT_SLEEP

mltrain_api = Blueprint('mltrain_api', __name__)


@mltrain_api.route("/mltrain", methods=["POST"])
def mltrain():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    
    try:
        assert request.method == 'POST'
        assert request.data
        assert request.is_json

        req_data = request.get_json()
        req_data=json.loads(req_data)
        # generate an ID for the classification then add the
        # classification ID + documents to the queue
        k = str(uuid.uuid4())
        d = {"id": k,  "documents": req_data['testo'], "cap_maj_master": req_data['cap_maj_master'], "filtro": req_data['filtro'], "analisi": req_data['analisi'], "kfold": req_data['kfold'], "modello": req_data['modello']} #, "classe": req_data['classe']}
        print(json.dumps(d, indent=4))
        print("-------------------------------------")
        main.db.rpush(ML_TRAIN_QUEUE, json.dumps(d))
        #print(db.lrange(MLP_QUEUE,0,-1))
        while True:
            #print('Enter in API loop..')
            output=main.db.get(k)
            if output is not None:
                output.decode("utf-8")
                data['trained']=json.loads(output)
                #print(data)
                #delete the output from database and break from the polling loop
                main.db.delete(k)
                break
            time.sleep(CLIENT_SLEEP)
        data['success']=True
        return flask.jsonify(data)
       
    except KeyError as e:
        data["errore"]=e.message
        time.sleep(CLIENT_SLEEP)
        #rc = request.status.HTTP_400_BAD_REQUEST
        return flask.jsonify(data)
