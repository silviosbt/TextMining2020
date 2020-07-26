'''
Created on 26 lug 2020

@author: silvio Fabi
'''

import os, sys, time
import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle, json
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

# SKLEARN LIBRERIES
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# CUSTOM MODULES
from script.pre_processing import preProcessing
from script.plot_confusion_matrix import plot_confusion_matrix
from script.select_model import selectModel



def run_ml_cv(args, macroTopic):
    MODEL="BOW_{}".format(args['model'])
    BASE_DIR=args['basedir']
    FEATURE_SELECTION=args['feature_selection']
    DATASET=args['dataset'][0]
    PLOT_CM=args['plot_cm']
    max_words=args['max_words']
    kfold=10
    SEP=','
    
    ### INIT OUTPUT CONTAINER TYPE
    results=pd.DataFrame()
    cvscores=[]
    prec_macro_scores=[]
    recall_macro_scores=[]
    fscore_macro_scores=[]
    
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
    X_train, y_train, class_dic, ordina = preProcessing(documents)
    NCLASSI=int(len(class_dic))
    
    ############### START ELABORATION ########################
    time_start = time.time()

    ### SET MODELs, INPUT SHAPE and OUTPUT DIRECTORY
    print('start Cross Validation elaboration')
    piperun, output_dir, set_model  =  selectModel(MODEL,NCLASSI,FEATURE_SELECTION,BASE_DIR,max_words,stop_words,random_state)

    print('output directory: {}'.format(output_dir))
    print('training model: {}'.format(set_model))
    
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
        #piperun, output_dir, set_model  =  selectModel(MODEL,NCLASSI,FEATURE_SELECTION,BASE_DIR,stop_words,stop_words,random_state)
        #print('output directory: {}'.format(output_dir))
        #print('training model: {}'.format(set_model))
                    
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

        # COMPUTE PRECISION, RECALL E F1 SCORE METRICS FOR CLASSES    
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        # COMPUTE PRECISION, RECALL E F1 SCORE METRICS FOR FOLDS   
        prec_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')        
        # SAVE SCORES IN ASSIGNED LIST 
        prec_macro_scores.append(round(prec_macro*100,1))
        recall_macro_scores.append(round(recall_macro*100,1))
        fscore_macro_scores.append(round(fscore_macro*100,1))
        print('Print scores fold {}'.format(k))
        print('precision weighted: ' , prec_macro)
        print('recall weighted: ' , recall_macro)
        print('F1score weighted: ' , fscore_macro)
        
        # SAVE ALL CALSSES METRICS FOR EVERY FOLD IN A DATAFRAME 
        df=pd.DataFrame()
        i=range(NCLASSI)
        df=pd.DataFrame({'CLASSE' : pd.Series(ordina, index=i) , 'PRECISION' : pd.Series(precision, index=i).round(3),'RECALL' : pd.Series(recall, index = i).round(3), 'F1SCORE' : pd.Series(fscore, index = i).round(3)}) # 'AUC' : pd.Series(roc_auc, index = i).round(2)})
        results=results.append(df, ignore_index=True, sort=False)
        print('---------FINE ------ CICLO {0} ---------------' .format(k))
    
    foldpath=os.path.join(output_dir, "totale")
    if not os.path.exists(foldpath):
        os.makedirs(foldpath)
          
    # PLOT CONFUSION MATRIX
    if PLOT_CM:
        #cnf_matrix = confusion_matrix(y_test, y_pred)
        cm_tot=np.rint(cm_tot/(k*2))
        np.set_printoptions(precision=2)
        #Plot non-normalized confusion matrix
        plt.figure(6)
        plot_confusion_matrix(cm_tot.astype(int), classes=ordina, output=output_dir)    
            
    ## RISULTATI OTTENUTI PER OGNI CLASSE IN CROSS VALIDATION (MEDIA E DEVIAZIONE STANDARD)
    results['CLASSE']=results.CLASSE.astype(str)
    results['PRECISION']=results.PRECISION.astype(float)
    results['RECALL']=results.RECALL.astype(float)
    results['F1SCORE']=results.F1SCORE.astype(float)
    #results_tfidf['AUC']=results_tfidf.AUC.astype(float)
    results.to_csv(os.path.join(foldpath,"{}_folds_classes_score.csv".format(set_model)), encoding='utf-8', sep=SEP)
    df1_mean=results.groupby('CLASSE', sort=False).mean().round(3)
    df2_std=results.groupby('CLASSE', sort=False).std().round(3)
    df1_mean = df1_mean.loc[:, ~df1_mean.columns.str.contains('^Unnamed')]
    df2_std = df2_std.loc[:, ~df2_std.columns.str.contains('^Unnamed')]
    
    results=pd.DataFrame()
    results['PRECISION_MEAN']=df1_mean.PRECISION
    results['PRECISION_STD']=df2_std.PRECISION
    results['RECALL_MEAN']=df1_mean.RECALL
    results['RECALL_STD']=df2_std.RECALL
    results['F1SCORE_MEAN']=df1_mean.F1SCORE
    results['F1SCORE_STD']=df2_std.F1SCORE
    ## ORDINA IL DATASET PER CLASSE
    results=results.assign(ordina=ordina)
    results['ordina']=results.ordina.astype(int)
    results=results.sort_values(by='ordina')
    results=results.drop(['ordina'], axis=1)
    ## AGGIUNGE LA DESCRIZIONE DELLE CLASSI
    results['TOPICS']=macroTopic
    #results=results.iloc[:,[0,5,1,2,3,4]]
    results.set_index('TOPICS')
    #idx=0
    #results=results.insert(loc=idx, column='TOPICS', value=macroTopic)
    print(results)
    results.to_csv(os.path.join(foldpath,"{}_cv_classes_scores.csv".format(set_model)), encoding='utf-8', sep=SEP)

    ## RISULATI PER FOLDS IN CROSS VALIDATION (MEDIANDO TUTTE LE CLASSI IN OGNI CICLO DI ELABORAZIONE)
    df_accuratezza_scores=pd.DataFrame()
    ind=range(len(cvscores))
    df_accuratezza_scores=pd.DataFrame({'Accuracy' : pd.Series(cvscores, index = ind).round(1), 
                                     'Precision': pd.Series(prec_macro_scores, index=ind).round(1), 
                                     'Recall': pd.Series(recall_macro_scores, index=ind).round(1), 
                                     'F1score': pd.Series(fscore_macro_scores, index=ind).round(1)})
    df_accuratezza_scores.to_csv(os.path.join(foldpath,"{}_cv_folds.csv".format(set_model)), encoding='utf-8', sep=SEP)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    ## RISULTATI TOTALE (MEDIANDO TUTTI I FOLDS)
    df_totale_scores=pd.DataFrame([["{}".format(set_model), np.mean(cvscores).round(1),  np.std(cvscores).round(1), np.max(cvscores).round(1), 
                                                            np.mean(prec_macro_scores).round(1), np.std(prec_macro_scores).round(1), np.max(prec_macro_scores).round(1), 
                                                            np.mean(recall_macro_scores).round(1), np.std(recall_macro_scores).round(1), np.max(recall_macro_scores).round(1),
                                                            np.mean(fscore_macro_scores).round(1), np.std(fscore_macro_scores).round(1), np.max(fscore_macro_scores).round(1)]], 
                                                            columns=['Modello','accuracy_mean','accuracy_std','accuracy_max','precision_mean','precision_std','precision_max',
                                                                     'recall_mean','recall_std','recall_max','f1score_mean','f1score_std','f1score_max'])
    df_totale_scores.to_csv(os.path.join(foldpath,"{}_cv_total.csv".format(set_model)), encoding='utf-8', sep=SEP)

    ## SAVE TIME ELAPSED 
    time_elapsed = (time.time() - time_start)
    time_elapsed=time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
    f = open(os.path.join(foldpath, "timelapsed.txt"), "w")
    f.write("Time Elapsed: {0}".format(time_elapsed))
    f.close()    

     
     

