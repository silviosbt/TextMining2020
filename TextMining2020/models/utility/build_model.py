'''
Created on 11 lug 2020

@author: Utente
'''

from __future__ import print_function, division
import warnings
warnings.filterwarnings('ignore')
import os, sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle
# import sklearn libreries
from sklearn import svm
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
# import keras libreries 
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input,  Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, Sequential

# CUSTOM Modules
import main
from config.websetting import *

def create_sklearn_modelSVC(random_state):
    model =svm.LinearSVC(C=10000,random_state=random_state)
    return model

def create_sklearn_modelCNB(random_state):
    model = ComplementNB(alpha=0.2)
    return model

def create_sklearn_modelPAC(random_state):
    model = PassiveAggressiveClassifier(C=1.0, max_iter=1000, early_stopping=True, validation_fraction=main.VALIDATION_SPLIT, n_iter_no_change=4, random_state=random_state, tol=1e-3)
    return model
 
def create_sklearn_modelMLP(random_state):
    model=MLPClassifier(hidden_layer_sizes=[128,128], activation='relu', solver='adam', verbose=False, early_stopping=True, validation_fraction=main.VALIDATION_SPLIT, n_iter_no_change=10, random_state=random_state)
    return model

def create_modelMLP1(tokenizer):
    vocab_size = len(tokenizer.word_index)+1
    model = Sequential()
    #model.add(Dense(128, input_shape=(max_words,)))
    model.add(Dense(128, input_shape=(vocab_size, )))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(main.NCLASSI))
    model.add(Activation('softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_modelMLP2(tokenizer):
    vocab_size = len(tokenizer.word_index)+1   
    model = Sequential()
    #model.add(Dense(128, input_shape=(max_words,)))
    model.add(Dense(128, input_shape=(vocab_size, )))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dense(main.NCLASSI))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def use_keras_modelMLP(X_train,y_train,X_test,y_test,model,k=1):
    # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH TF-IDF INDEX    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_matrix(X_train, mode='tfidf')
    X_test = tokenizer.texts_to_matrix(X_test, mode='tfidf')
    y_train = to_categorical(y_train, num_classes=main.NCLASSI)
    y_test = to_categorical(y_test, num_classes=main.NCLASSI)
    print(y_test)
    print(type(y_test))
    #vocab_size = len(tokenizer.word_index)+1
    #num_classes=y_train.shape[1]
    print("Numero di classi:", y_train.shape[1])
    print('Shape of X train and X validation tensor:', X_train.shape,X_test.shape)
    print('Shape of label train and validation tensor:', y_train.shape,y_test.shape)
    print("-------")
    OUTPUT_DIR=os.path.join(main.BASE_DIR,model) 
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR) 
             
    # create fold path for results save
    foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
    if not os.path.exists(foldpath):
        os.makedirs(foldpath)
    
    #save tokenizer
    with open(os.path.join(foldpath,"tokenizer_fold"+str(k)+"_"+model+".pickle"), 'wb') as handle:
        pickle.dump(tokenizer,handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Create callbacks
    name_weights = os.path.join(foldpath, "model_fold" + str(k) + "_"+model+".hdf5")
    checkpointer = ModelCheckpoint(filepath=name_weights,  verbose=2, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
    callbacks_list = [early_stopping,checkpointer]
    
    # build model
    model = create_modelMLP1()

    # fit network
    #history=model.fit(X_train, y_train, batch_size=BATCH, validation_data=(X_test, y_test), callbacks=callbacks_list, epochs=EPOCHS, verbose=VERBOSE_FIT)
    history = model.fit(X_train, y_train, batch_size=main.BATCH, epochs=main.EPOCHS, verbose=main.VERBOSE_FIT, callbacks=callbacks_list, validation_split=main.VALIDATION_SPLIT)
            
    print("Validation Loss: %.2f%%  (+/- %.2f%%)" % (np.mean(history.history['val_loss']), np.std(history.history['val_loss'])))
    print("Validation Accuracy: %.2f%%  (+/- %.2f%%)" % (np.mean(history.history['accuracy']), np.std(history.history['accuracy'])))
    return model 


def setEmbeddingMatrix(embedding, word_index):
    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    wordsembedding = []
    #nb_words = min(MAX_NB_WORDS, len(word_index))
    vocabulary_size=len(word_index) + 1
    embedding_matrix = np.zeros((vocabulary_size, main.EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= vocabulary_size:
            continue
        embedding_vector = embedding.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    del(embedding)
    print('Shape of embedding weight matrix: ', embedding_matrix.shape)
    nonzero_words=np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print('number of  word embeddings: %d' % nonzero_words)
    wordsembedding.append(nonzero_words)
    return embedding_matrix
    
def create_keras_modelCNN(raw_embedding,tokenizer):
    embedding_matrix=setEmbeddingMatrix(raw_embedding,tokenizer.word_index)
    print('Training model.')
    # define model,  train a 1D convnet
    sequence_input = Input(shape=(main.MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = Embedding(sequence_input, main.EMBEDDING_DIM, weights=[embedding_matrix], input_length=main.MAX_SEQUENCE_LENGTH, trainable=False)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    #x = GlobalMaxPooling1D()(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    #x = Dropout(0.5)(x)
    preds = Dense(main.NCLASSI, activation='softmax')(x)
    model = Model(sequence_input, preds)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model