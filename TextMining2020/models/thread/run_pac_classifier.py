'''
Created on 04 lug 2020

@author: Utente
'''

import pickle
import warnings
warnings.filterwarnings('ignore')
import main
from models.thread.runModel import runModel

# PAC Classifier without Feature Selection
def pac_classifier(target_names):
    
    print("* Loading PAC Model...")
    with open(main.MODELLO_PAC, 'rb') as file:
        model_pac=pickle.load(file)
    print("* Model PAC loaded")
    # continually pool for new  text to classify
    runModel(model_pac, main.PAC_QUEUE, target_names)

            
# PAC Classifier without Feature Selection
def pacfs_classifier(target_names):
    
    print("* Loading PAC Feature Selection Model...")
    with open(main.MODELLO_PAC_FS, 'rb') as file:
        model_pacfs=pickle.load(file)
    print("* Model PAC Feature Selection loaded")
    # continually pool for new  text to classify
    runModel(model_pacfs, main.PACFS_QUEUE, target_names)
