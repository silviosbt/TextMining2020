'''
Created on 04 lug 2020

@author: Utente
'''

import warnings
warnings.filterwarnings('ignore')
import flask
import redis
import os, time, json
from threading import Thread
# APPLICATION PACKEGES
from config.websetting import *
from models.thread import run_mlp_classifier, run_cnb_classifier, run_svm_classifier, run_pac_classifier, run_dl_classifier, run_ml_trainer
from api.rest.endpoint.svm_api import svm_api
from api.rest.endpoint.mlp_api import mlp_api
from api.rest.endpoint.cnb_api import cnb_api
from api.rest.endpoint.pac_api import pac_api
from api.rest.endpoint.deep_api import deep_api
from api.rest.endpoint.ml_trainer_api import mltrain_api


# SET TARGET DICTIONARY
def loadTarget(TARGET):
    print("* Loading TARGET dictionary...")
    with open(TARGET, 'r') as json_file:
        target_dict = json.load(json_file)
    print("* TARGET dictionary loaded")
    target_names=[x for x in target_dict.keys()]
    return target_names 

# initialize  Flask application, Redis server, and classifier model
app = flask.Flask(__name__)
db = redis.StrictRedis(host=REDISDB, port=6379, db=0)
db.flushall()

# load configuration setting from file setting.cfg
app.config.from_pyfile(os.path.join('.', 'config/setting.cfg'), silent=True)
BASE_DIR = app.config['BASE_DIR']
TARGET = app.config['TARGET']
#TARGET={"1": 0, "12": 1, "18": 2, "2": 3, "9": 4, "20": 5, "15": 6, "21": 7, "7": 8, "16": 9, "3": 10, "19": 11, "5": 12, "4": 13, "17": 14, "6": 15, "10": 16, "13": 17, "14": 18, "8": 19, "23": 20}
MODELLO_SVM=app.config['MODELLO_SVM']
MODELLO_SVM_FS=app.config['MODELLO_SVM_FS']
MODELLO_MLP=app.config['MODELLO_MLP']
MODELLO_MLP_FS=app.config['MODELLO_MLP_FS']
MODELLO_CNB=app.config['MODELLO_CNB']
MODELLO_CNB_FS=app.config['MODELLO_CNB_FS']
MODELLO_PAC=app.config['MODELLO_PAC']
MODELLO_PAC_FS=app.config['MODELLO_PAC_FS']
MODELLO_DL=app.config['MODELLO_DL']
TOKENIZER_DL=app.config['TOKENIZER_DL']
MAX_SEQUENCE_LENGTH=app.config['MAX_SEQUENCE_LENGTH']
MAX_WORDS=app.config['MAX_WORDS']
PLOT_CM=app.config['PLOT_CM']
NCLASSI=app.config['NCLASSI']
VALIDATION_SPLIT=app.config['VALIDATION_SPLIT']
EPOCHS=app.config['EPOCHS']
BATCH=app.config['BATCH']
VERBOSE_FIT=app.config['VERBOSE_FIT']
EMBEDDING_DIM=app.config['EMBEDDING_DIM']
## REGISTER APIs REST INTERFACE
app.register_blueprint(svm_api)
app.register_blueprint(mlp_api)
app.register_blueprint(cnb_api)
app.register_blueprint(pac_api)
app.register_blueprint(deep_api)
app.register_blueprint(mltrain_api)
#######

# The main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    # Load target dictionary (labels encoding)
    target_names=loadTarget(TARGET)
    # Load the function used to classify input documents in a *separate*
    # thread than the one used for main classification
    print("* Starting model mlp service without features selection...")
    t = Thread(target=run_mlp_classifier.mlp_classifier, args=(target_names,))
    t.daemon = True
    t.start()
    
    print("* Starting model mlp service with features selection...")
    z = Thread(target=run_mlp_classifier.mlpfs_classifier, args=(target_names,))
    z.daemon = True
    z.start()
    
    print("* Starting model svm service without features selection...")
    x = Thread(target=run_svm_classifier.svm_classifier, args=(target_names,))
    x.daemon = True
    x.start()
    
    print("* Starting model svm service with features selection...")
    y = Thread(target=run_svm_classifier.svmfs_classifier, args=(target_names,))
    y.daemon = True
    y.start()
    
    print("* Starting model cnb service without features selection...")
    o = Thread(target=run_cnb_classifier.cnb_classifier, args=(target_names,))
    o.daemon = True
    o.start()
    
    print("* Starting model cnb service with features selection...")
    p = Thread(target=run_cnb_classifier.cnbfs_classifier, args=(target_names,))
    p.daemon = True
    p.start()
    
    print("* Starting model pac service without features selection...")
    w = Thread(target=run_pac_classifier.pac_classifier, args=(target_names,))
    w.daemon = True
    w.start()
    
    print("* Starting model pac service with features selection...")
    j = Thread(target=run_pac_classifier.pacfs_classifier, args=(target_names,))
    j.daemon = True
    j.start()
    
    print("* Starting model deep learning service...")
    d = Thread(target=run_dl_classifier.dl_classifier, args=(target_names,))
    d.daemon = True
    d.start()
    
    print("* Starting model deep learning service...")
    ml = Thread(target=run_ml_trainer.ml_trainer, args=())
    ml.daemon = True
    ml.start()
    
    time.sleep(SERVER_SLEEP)    
    # start the web server
    print("* Starting web service...")
    #app.run()
    app.run( host='127.0.0.1', port=8085, debug=False)
