'''
Created on 08 lug 2020

@author: Utente
'''

# import the necessary packages
import warnings
warnings.filterwarnings('ignore')
import os, time
import pandas as pd
import numpy as np
import  pickle,json
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from itertools import chain

# import sklearn libreries
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# CUSTOM Modules
import main
from config.websetting import ML_TRAIN_QUEUE, SERVER_SLEEP
from models.utility.pre_processing import preProcessing
from models.utility.plot_confusion_matrix import plot_confusion_matrix
from models.utility.build_model import create_sklearn_modelSVC
from models.utility.build_model import create_sklearn_modelCNB
from models.utility.build_model import create_sklearn_modelPAC
from models.utility.build_model import create_sklearn_modelMLP


def selectModel(modello,filtro,stop_words,random_state):
    EMBEDDING=modello
    FEATURE_SELECTION=filtro
    STOP_WORDS=stop_words
    score_func=mutual_info_classif
    ### SET MODELs, INPUT SHAPE and OUTPUT DIRECTORY 
    if EMBEDDING=='BOW_SVM' or EMBEDDING=='BOW_{}_SVM'.format(main.MAX_WORDS):
        model = create_sklearn_modelSVC(random_state)
        if FEATURE_SELECTION == 'True':
            EMBEDDING='BOW_{}_SVM'.format(main.MAX_WORDS)
            OUTPUT_DIR=os.path.join(main.BASE_DIR, EMBEDDING)
            # create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=STOP_WORDS)
            kbestselect = SelectKBest(score_func=score_func,k=main.MAX_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
        else:
            OUTPUT_DIR=os.path.join(main.BASE_DIR, EMBEDDING)
            # create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=STOP_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('tfidf',vectorizer),('model', model)])
    elif EMBEDDING=='BOW_CNB' or EMBEDDING=='BOW_{}_CNB'.format(main.MAX_WORDS):
        model = create_sklearn_modelCNB(random_state)
        if FEATURE_SELECTION == 'True':
            EMBEDDING='BOW_{}_CNB'.format(main.MAX_WORDS)
            OUTPUT_DIR=os.path.join(main.BASE_DIR, EMBEDDING)
            # create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=STOP_WORDS)
            kbestselect = SelectKBest(score_func=score_func,k=main.MAX_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
        else:
            OUTPUT_DIR=os.path.join(main.BASE_DIR, EMBEDDING)
            # create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=STOP_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('tfidf',vectorizer),('model', model)])
    elif EMBEDDING=='BOW_PAC' or EMBEDDING=='BOW_{}_PAC'.format(main.MAX_WORDS):
        model = create_sklearn_modelPAC(random_state)
        #model = PassiveAggressiveClassifier(C=0.001, max_iter=1000, random_state=random_state)
        if FEATURE_SELECTION == 'True':
            EMBEDDING='BOW_{}_PAC'.format(main.MAX_WORDS)
            OUTPUT_DIR=os.path.join(main.BASE_DIR, EMBEDDING)
            # Create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=STOP_WORDS)
            kbestselect = SelectKBest(score_func=score_func,k=main.MAX_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
        else:
            OUTPUT_DIR=os.path.join(main.BASE_DIR, EMBEDDING)
            # Create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=STOP_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('tfidf',vectorizer),('model', model)])
    elif EMBEDDING=='BOW_MLP' or EMBEDDING=='BOW_{}_MLP'.format(main.MAX_WORDS):
        model=create_sklearn_modelMLP()
        if FEATURE_SELECTION == 'True':
            EMBEDDING='BOW_{}_MLP'.format(main.MAX_WORDS)
            OUTPUT_DIR=os.path.join(main.BASE_DIR, EMBEDDING)
            # Create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=STOP_WORDS)
            kbestselect = SelectKBest(score_func=score_func,k=main.MAX_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('cvectorizer',cvectorizer),('kbestselect',kbestselect),('model', model)])
        else:
            OUTPUT_DIR=os.path.join(main.BASE_DIR, EMBEDDING)
            # Create path for results save
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            # BUILD TFIDF BAG OF WORDS TEXT RAPPRESENTATION WITHOUT FEATURE SELECTION
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=STOP_WORDS)
            # SET PIPELINE
            pipe = Pipeline([('tfidf',vectorizer),('model', model)])   
    
    return pipe, OUTPUT_DIR, EMBEDDING


# Machine Learning models training
def ml_trainer():
    while True:
        # attempt to grab a json of documents from the database
        #queue = db.lrange(MLP_QUEUE, 0, BATCH_SIZE - 1)
        q = main.db.lpop(ML_TRAIN_QUEUE)
        if q is not None:
            #deserialize the object for get the data
            q=json.loads(q.decode("utf-8"))
            #print(type(q))
            print("-------------------------------------")
            items=q['documents']
            labels=q['cap_maj_master']
            filtro=q['filtro']
            analisi=q['analisi']
            kfold=int(q['kfold'])
            modello=q['modello']
            # build dataframe to train by the model
            i=range(len(items))            
            #documents=pd.DataFrame({'testo' : pd.Series(items, index=i) , 'labels' : pd.Series(labels, index=i).fillna(0).astype(int)})
            documents=pd.DataFrame({'testo' : pd.Series(items, index=i) , 'labels' : pd.Series(labels, index=i)})
            
            # set document ID (docIDs)
            docIDs=q['id']
            print('Numero di classi: ', main.NCLASSI)
            # fix random seed for reproducibility
            seed = 777
            random_state=seed
            stop_words = set(stopwords.words('italian'))
            # temporany lists for data store
            results_tfidf=pd.DataFrame()
            cvscores=[]
            prec_macro_scores=[]
            recall_macro_scores=[]
            fscore_macro_scores=[]
            time_start = time.time()
            X_train, y_train, class_dic, ordina = preProcessing(documents)
            target_names = ["classe {}".format(x) for x in class_dic.keys()]
            if not analisi:
                print('entro nel blocco not analisi')
                piperun, output_dir, set_model  =  selectModel(modello,filtro,stop_words,random_state)
                
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
            else:
                print('entro nel blocco analisi')
                data_corpus=X_train
                labels=y_train
                cm_tot=0
                k=0
                
                skf = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=2, random_state=random_state)
                for train_index, test_index in skf.split(data_corpus, labels):    
                    k+=1
                    #if k>2:  continue
                    print('--------INIZIO --- CICLO {0} ---------------' .format(k))
                    X_train, X_test = data_corpus.iloc[train_index].values, data_corpus.iloc[test_index].values
                    y_train, y_test = labels[train_index], labels[test_index]
                    print('Shape of X train and X validation tensor:', X_train.shape,X_test.shape)
                    print('Shape of label train and validation tensor:', y_train.shape,y_test.shape)
                    print("-------")
                    ### BUILD MODEL
                    piperun, output_dir, set_model  =  selectModel(modello,filtro,stop_words,random_state)
                    
                    ### FIT MODEL
                    print('FIT PIPELINE')
                    piperun.fit(X_train, y_train)
                    
                    foldpath=os.path.join(output_dir, "fold" + str(k))
                    if not os.path.exists(foldpath):
                        os.makedirs(foldpath)

                    ## SAVE MODEL
                    print('SAVE PIPELINE')
                    with open(os.path.join(foldpath, "model_"+set_model+".pickle"), 'wb') as handle:
                        pickle.dump(piperun, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    ## SAVE TARGET DICTIONARY
                    print('SAVE TARGET DICTIONARY')     
                    with open(os.path.join(foldpath,"target.json"), "w") as outfile:  
                        json.dump(class_dic, outfile)
                    
                    ### PREDICTION
                    y_pred=piperun.predict(X_test)
                    
                    ### EVALUATION
                    score=accuracy_score(y_test, y_pred) * 100
                    print(" Ciclo %d Accuracy: %.2f%%" % (k, score))
                    cvscores.append(score)                
                    
                    # COMPUTE CONFUSION MATRIX  
                    cm=confusion_matrix(y_test, y_pred)
                    cm_tot=cm_tot + cm
                    #print(classification_report(y_test, y_pred)) 

                    # COMPUTE PRECISION, RECALL E F1 SCORE METRICS    
                    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
                    prec_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                    prec_macro_scores.append(prec_macro*100)
                    recall_macro_scores.append(recall_macro*100)
                    fscore_macro_scores.append(fscore_macro*100)
                    print('precision macro: ' , prec_macro)
                    print('recall macro: ' , recall_macro)
                    print('F1score macro: ' , fscore_macro)
                    # SAVE ALL CALSSES METRICS FOR EVERY FOLD IN A DATAFRAME 
                    df=pd.DataFrame()
                    i=range(main.NCLASSI)
                    df=pd.DataFrame({'CLASSE' : pd.Series(target_names, index=i) , 'PRECISION' : pd.Series(precision, index=i).round(2),'RECALL' : pd.Series(recall, index = i).round(2), 'F1SCORE' : pd.Series(fscore, index = i).round(2)}) # 'AUC' : pd.Series(roc_auc, index = i).round(2)})
                    results_tfidf=results_tfidf.append(df, ignore_index=True, sort=False)
                    print('---------FINE ------ CICLO {0} ---------------' .format(k))
                
                # Compute confusion matrix
                if main.PLOT_CM=='True':
                    #cnf_matrix = confusion_matrix(y_test, y_pred)
                    cm_tot=np.rint(cm_tot/(k*2))
                    np.set_printoptions(precision=2)
                    #Plot non-normalized confusion matrix
                    plt.figure(6)
                    plot_confusion_matrix(cm_tot.astype(int), classes=ordina, output=output_dir, title='Confusion matrix')    
            
                dfdoc={}        
                dftotale={}
                dfclass={}
                results_tfidf['CLASSE']=results_tfidf.CLASSE.astype(str)
                results_tfidf['PRECISION']=results_tfidf.PRECISION.astype(float)
                results_tfidf['RECALL']=results_tfidf.RECALL.astype(float)
                results_tfidf['F1SCORE']=results_tfidf.F1SCORE.astype(float)
                results_tfidf=results_tfidf.groupby('CLASSE', sort=False).mean().round(2)
                results_tfidf = results_tfidf.loc[:, ~results_tfidf.columns.str.contains('^Unnamed')]
                results_tfidf=results_tfidf.assign(ordina=ordina)
                results_tfidf['ordina']=results_tfidf.ordina.astype(int)
                results_tfidf=results_tfidf.sort_values(by='ordina')
                #results_tfidf=results_tfidf.drop(['ordina'], axis=1)
                
                #print(results_tfidf.index.values)
                #dfclass={'Classe': results_tfidf.index.values.tolist(), 'PPV': results_tfidf['PRECISION'].tolist(), 'TPR': results_tfidf['RECALL'].tolist(), 
                #         'Fmeasure': results_tfidf['F1SCORE'].tolist()}
                dfclass={'Classe': results_tfidf['ordina'].tolist(), 'PPV': results_tfidf['PRECISION'].tolist(), 'TPR': results_tfidf['RECALL'].tolist(), 
                         'Fmeasure': results_tfidf['F1SCORE'].tolist()}
                
                
                df=pd.DataFrame()
                ind=range(len(cvscores))
                df=pd.DataFrame({'Accuracy' : pd.Series(cvscores, index = ind).round(1), 'Precision': pd.Series(prec_macro_scores, index=ind).round(1), 
                                 'Recall': pd.Series(recall_macro_scores, index=ind).round(1), 'F1score': pd.Series(fscore_macro_scores, index=ind).round(1)})
                dfdoc={'Accuratezza': df.Accuracy.tolist(), 'Precisione': df.Precision.tolist(), 'Recall': df.Recall.tolist(), 'F1score': df.F1score.tolist()}
                #doc={'codice': documents['id'].tolist(), "testo": documents['testo'].tolist(), "classe": documents['cap_maj_master'].tolist()}
                #df_accuratezza_scores.to_csv(os.path.join(foldpath,"accuratezza_"+modello+"_folds.csv"), encoding='utf-8' )
                df=pd.DataFrame()
                df=pd.DataFrame([[set_model, np.mean(cvscores).round(1),  np.std(cvscores).round(1), np.max(cvscores).round(1), np.mean(prec_macro_scores).round(1), 
                                  np.mean(recall_macro_scores).round(1), np.mean(fscore_macro_scores).round(1)]], 
                                  columns=['Modello','accuracy_mean','accuracy_std','accuracy_max','precision','recall','f1score'])
                #dftotale={'modello': modello, 'Accuratezza media': df.accuracy_mean.value(), 'Accuratezza std': df.accuracy_std.value(), 'Accuratezza max': df.accuracy_max.value(), 'Precisione media': df.precision.value(), 'Recall media': df.recall.value(), 'F1score media': df.f1score.value()}
                dftotale={'modello': set_model, 'Accuracy mean': np.mean(cvscores).round(1),  'Accuracy std': np.std(cvscores).round(1), 'Accuracy max':  np.max(cvscores).round(1), \
                          'Precision mean': np.mean(prec_macro_scores).round(1), 'Precision std': np.std(prec_macro_scores).round(1), 'Precision max': np.max(prec_macro_scores).round(1), \
                          'Recall mean': np.mean(recall_macro_scores).round(1), 'Recall std': np.std(recall_macro_scores).round(1), 'Recall max': np.max(recall_macro_scores).round(1), \
                          'F1score mean': np.mean(fscore_macro_scores).round(1), 'F1score std': np.std(fscore_macro_scores).round(1), 'F1score max': np.max(fscore_macro_scores).round(1)}
                #df_totale_scores.to_csv(os.path.join(foldpath,"bow+"+modello+"_final.csv"), encoding='utf-8' )
                output=dict(chain.from_iterable(d.items() for d in (dfdoc, dftotale)))
                output=dict(chain.from_iterable(d.items() for d in (dfclass, output)))
                #r = {"did": cod, "classe": labels}        
            print(output)
            print(type(output))
            print(json.dumps(output, indent=4))
            # store the output predictions in the database, using
            # the docIDs as the key so we can fetch the results
            main.db.set(docIDs, json.dumps(output))
            # sleep for a small amount
            time.sleep(SERVER_SLEEP)
            # remove the docIDs from our queue
            #db.ltrim(MLP_QUEUE, len(docIDs), -1)
            #db.lpop(docIDs)
            main.db.delete(docIDs)
            