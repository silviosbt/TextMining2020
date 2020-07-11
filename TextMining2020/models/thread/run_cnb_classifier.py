'''
Created on 04 lug 2020

@author: Utente
'''

import pickle
import warnings
warnings.filterwarnings('ignore')
import main
from models.thread.runModel import runModel

# CNB Classifier without Feature Selection
def cnb_classifier(target_names):
    
    print("* Loading CNB Model...")
    with open(main.MODELLO_CNB, 'rb') as file:
        model_cnb=pickle.load(file)
    print("* Model CNB loaded")    
    # continually pool for new  text to classify
    runModel(model_cnb, main.CNB_QUEUE, target_names)    

            
# CNB Classifier without Feature Selection
def cnbfs_classifier(target_names):
    
    print("* Loading CNB Feature Selection Model...")
    with open(main.MODELLO_CNB_FS, 'rb') as file:
        model_cnbfs=pickle.load(file)
    print("* Model CNB Feature Selection loaded")
    # continually pool for new  text to classify
    runModel(model_cnbfs, main.CNBFS_QUEUE, target_names)    
