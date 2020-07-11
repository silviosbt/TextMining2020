'''
Created on 05 lug 2020

@author: Utente
'''
import warnings
warnings.filterwarnings('ignore')
import time
import os
import main
# Disabilita tensorfolw GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import pandas as pd
import numpy as np
import pickle,json
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def dl_classifier(target_names):
    # load the pre-trained Keras model
    print("* Loading Deep Learning model...")
    model_dl = load_model(main.MODELLO_DL)
    print("* Model Deep Learning loaded")

    print("* Loading Deep Learning Tokenizer")
    with open(main.TOKENIZER_DL, 'rb') as handle:
        tokenizer =pickle.load(handle)
    print("* Tokenizer Deep Learning loaded")

    # continually pool for new images to classify
    while True:
        # attempt to grab a json of documents from the database
        q = main.db.lpop(main.DEEP_QUEUE)
        if q is not None:
            #for qqq in queue:
            print('valore di q:',q)
            #deserialize the object for get the data
            q=json.loads(q.decode("utf-8"))
            print(type(q))
            print("-------------------------------")
            documents=q['documents']
            print("lunghezza lista documenti:", len(documents))
            codice=q['codice']
            print("Print Type documents:")
            print(type(documents))
            # Identificativo univoco del task, (chiave univoca)
            docIDs=q['id']
            X_test=documents
            print(type(X_test))
            # create BoW by tfidf
            #X_test_tfidf = tokenizer.texts_to_matrix(X_test)
            sequences = tokenizer.texts_to_sequences(X_test)
            sequences = pad_sequences(sequences,maxlen=main.MAX_SEQUENCE_LENGTH, padding='post')
            #print(sequences.shape)

            i=0
            output=[]
            for x_t in sequences:
                prediction = model_dl.predict(np.array([x_t]))
                predicted_label=prediction[0].argmax()
                print('Identificativo documento: {} -- Predicted label: {}'.format(codice[i], target_names[predicted_label]))
                r = {"did": codice[i], "testo": documents[i], "classe": target_names[predicted_label]}
                output.append(r)
                i += 1
            print(json.dumps(output, indent=4))
            # store the output predictions in the database, using
            # the docID as the key so we can fetch the results
            main.db.set(docIDs, json.dumps(output))
            # sleep for a small amount
            time.sleep(main.SERVER_SLEEP)
            # remove the docIDs from our queue
            #db.ltrim(MLP_QUEUE, len(docIDs), -1)
            #db.lpop(docIDs)
            main.db.delete(docIDs)