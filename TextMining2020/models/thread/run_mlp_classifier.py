'''
Created on 04 lug 2020

@author: Utente
'''

import pickle
import warnings
warnings.filterwarnings('ignore')
import main
from models.thread.runModel import runModel

# MLP Classifier without Feature Selection
def mlp_classifier(target_names):
    
    print("* Loading MLP Model...")
    with open(main.MODELLO_MLP, 'rb') as file:
        model_mlp=pickle.load(file)
    print("* Model MLP loaded")
    # continually pool for new  text to classify
    runModel(model_mlp, main.MLP_QUEUE, target_names)  

# MLP Classifier with Feature Selection
def mlpfs_classifier(target_names):
    
    print("* Loading MLP Feature Selection Model...")
    with open(main.MODELLO_MLP_FS, 'rb') as file:
        model_mlpfs=pickle.load(file)
    print("* Model MLP Feature Selection loaded")
    # continually pool for new  text to classify
    runModel(model_mlpfs, main.MLPFS_QUEUE, target_names)  
           