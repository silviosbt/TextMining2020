'''
Created on 11 lug 2020

@author: Utente
'''

import re
import pandas as pd
import numpy as np
# CUSTOM Modules
from models.utility.create_labels_dict import create_labels_dict


def preProcessing(documents):           
    documents=documents[pd.notnull(documents['labels'])]
    documents['labels']=documents['labels'].fillna(0).astype(int)
    #print(documents.head(5))
    #### PRE PROCESSING STEP. CLEAN OUTPUT TEXT DOCUMENTS
    # delete empty documents and documents with #N/A values in target column 
    # print(documents['cap_maj_master'].isnull().sum())
    documents=documents[pd.notnull(documents['labels'])]
    documents=documents[pd.notnull(documents['testo'])]
    # print(documents['cap_maj_master'].isnull().sum())
    print("Numero di documenti da processare: %d" %  len(documents))
    # remove parole ripetute e condivise tra la maggior parte dei documenti, remove numeri e altri caratteri
    regex_pat = re.compile(r'premesso|quale|\d+|\(\d+\)|:|,|-', flags=re.IGNORECASE)
    documents['testo']=documents['testo'].str.replace(regex_pat,'')
    X_train=documents.testo
    print('--------')
    print('print unique labels:', documents['labels'].unique())
    ordina=documents['labels'].unique()
    class_dic, labels=create_labels_dict(documents['labels'].astype(int).astype(str))
    y_train=np.asarray(labels).astype(int)
    return X_train, y_train, class_dic, ordina 