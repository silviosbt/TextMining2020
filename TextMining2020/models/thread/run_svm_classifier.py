'''
Created on 04 lug 2020

@author: Utente
'''

import pickle
import warnings
warnings.filterwarnings('ignore')
import main
from models.thread.runModel import runModel
 
# SVM Classifier without Feature Selection
def svm_classifier(target_names):
    
    print("* Loading SVM Model...")
    with open(main.MODELLO_SVM, 'rb') as file:
        model_svm=pickle.load(file)
    print("* Model SVM loaded")
    # continually pool for new  text to classify
    runModel(model_svm, main.SVM_QUEUE, target_names)
            
# SVM Classifier without Feature Selection
def svmfs_classifier(target_names):
    
    print("* Loading SVM Feature Selection Model...")
    with open(main.MODELLO_SVM_FS, 'rb') as file:
        model_svmfs=pickle.load(file)
    print("* Model SVM Feature Selection loaded")
    # continually pool for new  text to classify
    runModel(model_svmfs, main.SVMFS_QUEUE, target_names)
    
