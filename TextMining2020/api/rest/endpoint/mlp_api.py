'''
Created on 05 lug 2020

@author: Utente
'''


import flask
from flask import Blueprint, request
import json, time, uuid
import main 

mlp_api = Blueprint('mlp_api', __name__)

@mlp_api.route("/mlp", methods=["POST"])
def mlp():
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
        d = {"id": k, "codice": req_data['codice'], "documents": req_data['testo']} #, "classe": req_data['classe']}
        print(json.dumps(d, indent=4))
        print("-------------------------------------")
        main.db.rpush(main.MLP_QUEUE, json.dumps(d))
        #print(db.lrange(MLP_QUEUE,0,-1))
        while True:
            output=main.db.get(k)
            if output is not None:
                output.decode("utf-8")
                data['prediction']=json.loads(output)
                print(data)
                #delete the output from database and break from the polling loop
                main.db.delete(k)
                break
            time.sleep(main.CLIENT_SLEEP)
        data['success']=True
        return flask.jsonify(data)
       
    except KeyError as e:
        data["errore"]=e.message
        time.sleep(main.CLIENT_SLEEP)
        #rc = request.status.HTTP_400_BAD_REQUEST
        return flask.jsonify(data)

@mlp_api.route("/mlpfs", methods=["POST"])    
def mlpfs():
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
        d = {"id": k, "codice": req_data['codice'], "documents": req_data['testo']} #, "classe": req_data['classe']}
        print(json.dumps(d, indent=4))
        print("-------------------------------------")
        main.db.rpush(main.MLPFS_QUEUE, json.dumps(d))
        #print(db.lrange(MLP_QUEUE,0,-1))
        while True:
            output=main.db.get(k)
            if output is not None:
                output.decode("utf-8")
                data['prediction']=json.loads(output)
                print(data)
                #delete the output from database and break from the polling loop
                main.db.delete(k)
                break
            time.sleep(main.CLIENT_SLEEP)
        data['success']=True
        return flask.jsonify(data)
       
    except KeyError as e:
        data["errore"]=e.message
        time.sleep(main.CLIENT_SLEEP)
        #rc = request.status.HTTP_400_BAD_REQUEST
        return flask.jsonify(data)    
