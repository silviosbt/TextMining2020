'''
Created on 17 mag 2020

@author: Silvio Fabi
'''
#!/usr/bin/env LANG=en_US.UTF-8 /usr/local/bin/python3
from __future__ import print_function, division
import warnings
warnings.filterwarnings('ignore')
import os, sys, time, re
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
from models.utility.cmaps import cmaps 
cmaps=cmaps()
Blues=cmaps.Blues

def create_labels_dict(items):
    classi=items.unique()
    dic={}
    for i,classi in enumerate(classi):
        dic[classi]=i
    labels=items.apply(lambda x:dic[x])
    return dic, labels

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def create_sklearn_modelMLP():
    model=MLPClassifier(hidden_layer_sizes=[128,128], activation='relu', solver='adam', verbose=False, early_stopping=True, validation_fraction=VALIDATION_SPLIT, n_iter_no_change=10, random_state=random_state)
    return model

###########################  INZIO ELABORAZIONE ################################

#EMBEDDING= sys.argv[1]
EMBEDDING="BOW_CNB"
#BASE_DIR = '/home/silvio/datasets'
BASE_DIR = r'C:\\datasets'
DATASET=r'cap_qt_leg13_17.xlsx'
#DATASET=r'cap_qt_leg13_17.csv'
### SET MODEL PARAMETERS
max_words = 20000
score_func=mutual_info_classif
#score_func=chi2
FEATURE_SELECTION='false'
#FEATURE_SELECTION=sys.argv[2]
VALIDATION_SPLIT=0.2
encoding='utf-8'
#encoding='cp1252'

### UPLOAD CAP DATASET 
print('Carico Documenti da analizzare ')
#documents = pd.read_csv(os.path.join(BASE_DIR,DATASET), encoding='utf-8')
documents = pd.read_excel(os.path.join(BASE_DIR,DATASET), sheet_name="Foglio1", econding='utf-8')
print('Numero di documenti %s ' % len(documents))

# fix random seed for reproducibility
random_state=777
# set stop words list
stop_words = set(stopwords.words('italian'))

#### PRE PROCESSING STEP. CLEAN OUTPUT TEXT DOCUMENTS
# delete empty documents and documents with #N/A values in target column 
# print(documents['cap_maj_master'].isnull().sum())
documents=documents[pd.notnull(documents['cap_maj_master'])]
documents=documents[pd.notnull(documents['testo'])]
# print(documents['cap_maj_master'].isnull().sum())
#documents['cap_maj_master']=documents['cap_maj_master'].fillna(0).astype(int)
print("Numero di documenti da processare: %d" %  len(documents))
# remove parole ripetute e condivise tra la maggior parte dei documenti, remove numeri e altri caratteri
regex_pat = re.compile(r'premesso|quale|\d+|\(\d+\)|:|,|-', flags=re.IGNORECASE)
documents['testo']=documents['testo'].str.replace(regex_pat,'')
X_train=documents.testo
print('--------')

print('print unique labels:', documents['cap_maj_master'].unique())
ordina=documents['cap_maj_master'].unique()
class_dic, labels=create_labels_dict(documents['cap_maj_master'].astype(int).astype(str))
y_train=np.asarray(labels).astype(int)


############### START STRATIFIED CROSS VALIDATION WITH REPETITION ########################

time_start = time.time()

### SET MODELs, INPUT SHAPE and OUTPUT DIRECTORY 
if EMBEDDING=='BOW_SVM' or EMBEDDING=='BOW_{}_SVM'.format(max_words):
    model =svm.LinearSVC(C=10000,random_state=random_state)
    if FEATURE_SELECTION == 'True':
        EMBEDDING='BOW_{}_SVM'.format(max_words)
        OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
        # create path for results save
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
        cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=stop_words)
        kbestselect = SelectKBest(score_func=score_func,k=max_words)
        # SET PIPELINE
        pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
    else:
        OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
        # create path for results save
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
        vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=stop_words)
        # SET PIPELINE
        pipe = Pipeline([('tfidf',vectorizer),('model', model)])
        
elif EMBEDDING=='BOW_CNB' or EMBEDDING=='BOW_{}_CNB'.format(max_words):
    model = ComplementNB(alpha=0.2)
    if FEATURE_SELECTION == 'True':
        EMBEDDING='BOW_{}_CNB'.format(max_words)
        OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
        # create path for results save
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
        cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=stop_words)
        kbestselect = SelectKBest(score_func=score_func,k=max_words)
        # SET PIPELINE
        pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
    else:
        OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
        # create path for results save
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
        vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=stop_words)
        # SET PIPELINE
        pipe = Pipeline([('tfidf',vectorizer),('model', model)])

elif EMBEDDING=='BOW_PAC' or EMBEDDING=='BOW_{}_PAC'.format(max_words):
    model = PassiveAggressiveClassifier(C=1.0, max_iter=1000, early_stopping=True, validation_fraction=VALIDATION_SPLIT, n_iter_no_change=4, random_state=random_state, tol=1e-3)
    #model = PassiveAggressiveClassifier(C=0.001, max_iter=1000, random_state=random_state)
    if FEATURE_SELECTION == 'True':
        EMBEDDING='BOW_{}_PAC'.format(max_words)
        OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
        # Create path for results save
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
        cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=stop_words)
        kbestselect = SelectKBest(score_func=score_func,k=max_words)
        # SET PIPELINE
        pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
    else:
        OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
        # Create path for results save
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
        vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=stop_words)
        # SET PIPELINE
        pipe = Pipeline([('tfidf',vectorizer),('model', model)])
            
elif EMBEDDING=='BOW_MLP' or EMBEDDING=='BOW_{}_MLP'.format(max_words):
    model=create_sklearn_modelMLP()
    if FEATURE_SELECTION == 'True':
        EMBEDDING='BOW_{}_MLP'.format(max_words)
        OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
        # Create path for results save
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
        cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=stop_words)
        kbestselect = SelectKBest(score_func=score_func,k=max_words)
        # SET PIPELINE
        pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
    else:
        OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
        # Create path for results save
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
        vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=stop_words)
        # SET PIPELINE
        pipe = Pipeline([('tfidf',vectorizer),('model', model)])   
        
                
### FIT MODEL
print('FIT PIPELINE')
pipe.fit(X_train, y_train)

## SAVE MODEL
print('SAVE PIPELINE')
with open(os.path.join(OUTPUT_DIR, "model_"+EMBEDDING+".pickle"), 'wb') as handle:
    pickle.dump(pipe, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
## SAVE TARGET DICTIONARY     
with open(os.path.join(OUTPUT_DIR,"target.json"), "w") as outfile:  
    json.dump(class_dic, outfile)  
        
time_elapsed = (time.time() - time_start)
time_elapsed=time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
print("Time Elapsed: {0}".format(time_elapsed))
## TIME PROCESSIS
f = open(os.path.join(OUTPUT_DIR, "timelapsed.txt"), "w")
f.write("Time Elapsed: {0}".format(time_elapsed))
f.close()    

exit(0)