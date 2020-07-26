'''
Created on 19 lug 2020

@author: Utente
'''

# import the necessary packages
import warnings
warnings.filterwarnings('ignore')
import os 
# import sklearn libreries
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

# CUSTOM Modules
from script.build_model import create_sklearn_modelSVC
from script.build_model import create_sklearn_modelCNB
from script.build_model import create_sklearn_modelPAC
from script.build_model import create_sklearn_modelMLP


def selectModel(modello,num_classi,filtro,base_dir,max_words,stop_words,random_state):
    MODEL=modello
    NCLASSI=num_classi
    FEATURE_SELECTION=filtro
    BASE_DIR=base_dir
    MAX_WORDS=max_words
    STOP_WORDS=stop_words
    score_func=mutual_info_classif
    
    # FIX PARAMETERS
    VALIDATION_SPLIT=0.2
    encoding='utf-8'
    
    ### SET MODELs, INPUT SHAPE and OUTPUT DIRECTORY 
    if MODEL=='BOW_SVM' or MODEL=='BOW_SVM_{}'.format(MAX_WORDS):
        model = create_sklearn_modelSVC(random_state)
        if FEATURE_SELECTION:
            MODEL='BOW_SVM_{}'.format(MAX_WORDS)
            OUTPUT_DIR=os.path.join(BASE_DIR, MODEL)
            # create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=STOP_WORDS)
            kbestselect = SelectKBest(score_func=score_func,k=MAX_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
        else:
            OUTPUT_DIR=os.path.join(BASE_DIR, MODEL)
            # create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=STOP_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('tfidf',vectorizer),('model', model)])
    elif MODEL=='BOW_CNB' or MODEL=='BOW_CNB_{}'.format(MAX_WORDS):
        model = create_sklearn_modelCNB(random_state)
        if FEATURE_SELECTION:
            MODEL='BOW_CNB_{}'.format(MAX_WORDS)
            OUTPUT_DIR=os.path.join(BASE_DIR, MODEL)
            # create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=STOP_WORDS)
            kbestselect = SelectKBest(score_func=score_func,k=MAX_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
        else:
            OUTPUT_DIR=os.path.join(BASE_DIR, MODEL)
            # create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=STOP_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('tfidf',vectorizer),('model', model)])
    elif MODEL=='BOW_PAC' or MODEL=='BOW_PAC_{}'.format(MAX_WORDS):
        model = create_sklearn_modelPAC(random_state,VALIDATION_SPLIT)
        #model = PassiveAggressiveClassifier(C=0.001, max_iter=1000, random_state=random_state)
        if FEATURE_SELECTION:
            MODEL='BOW_PAC_{}'.format(MAX_WORDS)
            OUTPUT_DIR=os.path.join(BASE_DIR, MODEL)
            # Create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=STOP_WORDS)
            kbestselect = SelectKBest(score_func=score_func,k=MAX_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
        else:
            OUTPUT_DIR=os.path.join(BASE_DIR, MODEL)
            # Create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=STOP_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('tfidf',vectorizer),('model', model)])
    elif MODEL=='BOW_MLP' or MODEL=='BOW_MLP_{}'.format(MAX_WORDS):
        model=create_sklearn_modelMLP(random_state,VALIDATION_SPLIT)
        if FEATURE_SELECTION:
            MODEL='BOW_MLP_{}'.format(MAX_WORDS)
            OUTPUT_DIR=os.path.join(BASE_DIR, MODEL)
            # Create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=STOP_WORDS)
            kbestselect = SelectKBest(score_func=score_func,k=MAX_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
        else:
            OUTPUT_DIR=os.path.join(BASE_DIR, MODEL)
            # Create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=STOP_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('tfidf',vectorizer),('model', model)])   
    
    return pipe, OUTPUT_DIR, MODEL
