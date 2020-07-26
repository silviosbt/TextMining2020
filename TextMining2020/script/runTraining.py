'''
Created on 17 mag 2020

@author: Silvio Fabi
'''
import os, sys, time
import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle, json
from nltk.corpus import stopwords
# import sklearn libreries
from sklearn import svm
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

# CUSTOM Module
from script.pre_processing import preProcessing
from script.plot_confusion_matrix import plot_confusion_matrix
from script.select_model import selectModel



def run_ml_training(args):
    #MODEL= sys.argv[1]
    MODEL="BOW_{}".format(args['model'])
    BASE_DIR=args['basedir']
    FEATURE_SELECTION=args['feature_selection']
    DATASET=args['dataset'][0]
    max_words=args['max_words']

    ### UPLOAD CAP DATASET 
    print('Carico Documenti da analizzare ')
    filepath=DATASET
    if filepath.lower().endswith(('.xls', '.xlsx')):
        #xl = pd.ExcelFile(filepath, encoding='utf-8')
        xl = pd.ExcelFile(filepath)
        documents=xl.parse(xl.sheet_names[0])
        #documents = pd.read_excel(filepath, encoding='utf-8')
    else:
        documents = pd.read_csv(filepath, encoding='utf-8')
    print('Numero di documenti %s ' % len(documents))

    # fix random seed for reproducibility
    random_state=777
    # set stop words list
    stop_words = set(stopwords.words('italian'))

    #### PRE PROCESSING STEP. CLEAN OUTPUT TEXT DOCUMENTS
    X_train, y_train, class_dic, _ = preProcessing(documents)
    NCLASSI=int(len(class_dic))
    
    ############### START ELABORATION ########################
    time_start = time.time()

    ### SET MODELs, INPUT SHAPE and OUTPUT DIRECTORY
    print('START TRAINING JOB ...')
    piperun, output_dir, set_model  =  selectModel(MODEL,NCLASSI,FEATURE_SELECTION,BASE_DIR,max_words,stop_words,random_state)

    print('output directory: {}'.format(output_dir))
    print('modello da addestrare: {}'.format(set_model))
                
    ### FIT MODEL
    print('FIT PIPELINE')
    piperun.fit(X_train, y_train)

    ## SAVE MODEL
    print('SAVE PIPELINE')
    with open(os.path.join(output_dir, "model_"+set_model+".pickle"), 'wb') as handle:
        pickle.dump(piperun, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    ## SAVE TARGET DICTIONARY
    print('SAVE TARGET DICTIONARY')     
    with open(os.path.join(output_dir,"target.json"), "w") as outfile:  
        json.dump(class_dic, outfile)
                    
    time_elapsed = (time.time() - time_start)
    time_elapsed=time.strftime("%H:%M:%S", time.gmtime(time_elapsed)) 
    output={'modello': set_model, 'time_elapsed': time_elapsed, 'output_dir': output_dir}
    print('Training elapsed in: {}, model {} save in directory: {}'.format(output['time_elapsed'],output['modello'],output['output_dir']))
     
     
     
