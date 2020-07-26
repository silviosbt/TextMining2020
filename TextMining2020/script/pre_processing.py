'''
Created on 26 lug 2020

@author: Utente
'''

import re
import pandas as pd
import numpy as np


def create_labels_dict(items):
    classi=items.unique()
    dic={}
    for i,classi in enumerate(classi):
        dic[classi]=i
    labels=items.apply(lambda x:dic[x])
    return dic, labels

def preProcessing(documents):           
    documents=documents[pd.notnull(documents['cap_maj_master'])]
    documents['cap_maj_master']=documents['cap_maj_master'].fillna(0).astype(int)
    #print(documents.head(5))
    #### PRE PROCESSING STEP. CLEAN OUTPUT TEXT DOCUMENTS
    # delete empty documents and documents with #N/A values in target column 
    # print(documents['cap_maj_master'].isnull().sum())
    documents=documents[pd.notnull(documents['cap_maj_master'])]
    documents=documents[pd.notnull(documents['testo'])]
    # print(documents['cap_maj_master'].isnull().sum())
    print("Numero di documenti da processare: %d" %  len(documents))
    # remove repeat words along documents, remove digits and other characters
    regex_pat = re.compile(r'premesso|quale|\d+|\(\d+\)|:|,|-', flags=re.IGNORECASE)
    documents['testo']=documents['testo'].str.replace(regex_pat,'')
    X_train=documents.testo
    print('--------')
    print('print unique labels:', documents['cap_maj_master'].unique())
    ordina=documents['cap_maj_master'].unique()
    class_dic, labels=create_labels_dict(documents['cap_maj_master'].astype(int).astype(str))
    y_train=np.asarray(labels).astype(int)
    return X_train, y_train, class_dic, ordina 