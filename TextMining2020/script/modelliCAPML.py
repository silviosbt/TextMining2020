'''
Created on 17 mag 2020

@author: Silvio Fabi
'''
#!/usr/bin/env LANG=en_US.UTF-8 /usr/local/bin/python3
from __future__ import print_function, division
import warnings
warnings.filterwarnings('ignore')
import os, sys, codecs, time, re
import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import itertools
from operator import itemgetter
import pickle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
# import sklearn libreries
from sklearn import svm
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif 
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
# import keras libreries 
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input,  Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, Sequential

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

def plot_confusion_matrix(cmt, classes, normalize=False, title='Confusion matrix', cmap=Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cmt = cmt.astype('float') / cmt.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cmt, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    #plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cmt.max() / 2.
    for i, w in itertools.product(range(cmt.shape[0]), range(cmt.shape[1])):
        plt.text(w, i, format(cmt[i, w], fmt),
                 horizontalalignment="center",
                 color="white" if cmt[i, w] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.tight_layout()
    plotsave=os.path.join(foldpath,"confusion_matrix.{}".format(EXT) )
    plt.savefig(plotsave ,format=EXT, dpi=1000, bbox_inches='tight')    
    #plt.show()
    plt.close(6)
    
# load embedding into memory, skip first line from text file   
def load_embedding_text(filename):
    #f = open(filename,'rb')
    embeddings_index = {}
    f = codecs.open(filename,  encoding='utf-8')
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index    
 
# feature selection
def featureSelect(X_train,y_train,X_test,cvectorizer):
    # prepare vectorizer for select feature task 
    vectorizer = SelectKBest(score_func=score_func,k=max_words)
    X_train_best= vectorizer.fit_transform(X_train,y_train)
    X_test_best = vectorizer.transform(X_test)
    feature_names = cvectorizer.get_feature_names()
    mask = vectorizer.get_support() #list of booleans
    new_features = [] # The list of your K best features
    for boolv, feature in zip(mask, feature_names):
        if boolv:
            new_features.append(feature)
                
    res = dict(zip(new_features, vectorizer.scores_))
    topf=list(sorted(res.items(), key=itemgetter(1)))[::-1]

    features_ranking=[]
    topfeatures=[]
    for item in topf:
        if not item[0].isalpha() or len(item[0])<=3:
            continue
        else:
            topfeatures.append(item[0])
            features_ranking.append(" {} : {} ".format(item[0],item[1]))

    #print('FOLD_{0}> {1}'.format(k,features_ranking[:10]))            
    featureSelected['fold{}'.format(k)]=features_ranking[:10]
            
    with open(os.path.join(foldpath,"vectorizer_fold"+str(k)+"_"+EMBEDDING+".pickle"), 'wb') as handle:
        pickle.dump(vectorizer,handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    return X_train_best, X_test_best, feature_names

def top10_features_for_classes(feature_names):
    print("top 10 keywords per class:")
    feature_names = np.asarray(feature_names)
    for i, label in enumerate(target_names):
        top10 = np.argsort(model.coef_[i])[-10:]
        top10feature[label]=feature_names[top10]
    print()       
    f = open(os.path.join(foldpath, "top10features.txt"), "w", encoding='utf-8')
    f.write("Fold{0} feature: {1}".format(k, top10feature))
    f.close()
    ## BEST TOP 10 FEATURES SCORE FOR CKLASSES
    top10features=pd.DataFrame.from_dict(top10feature)
    top10features.to_csv(os.path.join(foldpath,"{}_top10features_score_classes.csv".format(EMBEDDING)), encoding='utf-8' ) 

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def create_sklearn_modelMLP():
    model=MLPClassifier(hidden_layer_sizes=[128,128], activation='relu', solver='adam', verbose=False, early_stopping=True, validation_fraction=VALIDATION_SPLIT, n_iter_no_change=10, random_state=random_state)
    return model

def create_modelMLP1():
    model = Sequential()
    #model.add(Dense(128, input_shape=(max_words,)))
    model.add(Dense(128, input_shape=(vocab_size, )))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NCLASSI))
    model.add(Activation('softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_modelMLP2():   
    model = Sequential()
    #model.add(Dense(128, input_shape=(max_words,)))
    model.add(Dense(128, input_shape=(vocab_size, )))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dense(NCLASSI))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def use_keras_modelMLP():
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH TF-IDF INDEX    
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(X_train)
            X_train = tokenizer.texts_to_matrix(X_train, mode='tfidf')
            X_test = tokenizer.texts_to_matrix(X_test, mode='tfidf')
            y_train = to_categorical(y_train, num_classes=NCLASSI)
            y_test = to_categorical(y_test, num_classes=NCLASSI)
            print(y_test)
            print(type(y_test))
            #vocab_size = len(tokenizer.word_index)+1
            #num_classes=y_train.shape[1]
            print("Numero di classi:", y_train.shape[1])
            print('Shape of X train and X validation tensor:', X_train.shape,X_test.shape)
            print('Shape of label train and validation tensor:', y_train.shape,y_test.shape)
            print("-------")
    
            # create fold path for results save
            foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
            if not os.path.exists(foldpath):
                os.makedirs(foldpath)
    
            #save tokenizer
            with open(os.path.join(foldpath,"tokenizer_fold"+str(k)+"_"+EMBEDDING+".pickle"), 'wb') as handle:
                pickle.dump(tokenizer,handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Create callbacks
            name_weights = os.path.join(foldpath, "model_fold" + str(k) + "_"+EMBEDDING+".hdf5")
            checkpointer = ModelCheckpoint(filepath=name_weights,  verbose=2, save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
            callbacks_list = [early_stopping,checkpointer]
    
            # genero il modello
            if MODELLOMLP==1:
                model = create_modelMLP1()
            else:
                model = create_modelMLP2()

            # fit network
            #history=model.fit(X_train, y_train, batch_size=BATCH, validation_data=(X_test, y_test), callbacks=callbacks_list, epochs=EPOCHS, verbose=VERBOSE_FIT)
            history = model.fit(X_train, y_train, batch_size=BATCH, epochs=EPOCHS, verbose=VERBOSE_FIT, callbacks=callbacks_list, validation_split=VALIDATION_SPLIT)
            
            print("Validation Loss: %.2f%%  (+/- %.2f%%)" % (np.mean(history.history['val_loss']), np.std(history.history['val_loss'])))
            print("Validation Accuracy: %.2f%%  (+/- %.2f%%)" % (np.mean(history.history['accuracy']), np.std(history.history['accuracy'])))



def setEmbeddingMatrix2(embedding, word_index):
    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    #nb_words = min(MAX_NB_WORDS, len(word_index))
    vocabulary_size=len(word_index) + 1
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
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

def create_model():
    print('Training model.')
    # define model,  train a 1D convnet
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
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
    preds = Dense(NCLASSI, activation='softmax')(x)
    model = Model(sequence_input, preds)
    print(model.summary())
    #plotsave=os.path.join(foldpath,"model_fold" + str(k) + "_schema.png" )
    #plot_model(model, to_file=plotsave, show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

 

###########################  INZIO ELABORAZIONE ################################

#EMBEDDING= sys.argv[1]
EMBEDDING="BOW_MLP"
#BASE_DIR = '/home/silvio/datasets'
BASE_DIR = r'C:\\datasets'
DATASET=r'cap_qt_leg13_17.xlsx'
#DATASET=r'cap_qt_leg13_17.csv'
### SET MODEL PARAMETERS
max_words = 20000
print_top10='False'
score_func=mutual_info_classif
#score_func=chi2
MAX_SEQUENCE_LENGTH=1500
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 300
VERBOSE_FIT=1
MODELLOMLP=2
FEATURE_SELECTION='True'
#FEATURE_SELECTION=sys.argv[2]
PLOT_CM='True'
EXT='png'
SEP=';'   #Separator for CSV file format
KNUM = 10 #Numero di fold da utilizzare. Deve essere almeno 4
EPOCHS=20
BATCH=128
DEEPMODEL=['WORD2VEC_CNN','GLOVE_CNN','FASTTEXT_CNN']
encoding='utf-8'
#encoding='cp1252'

### INIT OUTPUT CONTAINER TYPE
results_tfidf=pd.DataFrame()
cvscores=[]
prec_macro_scores=[]
recall_macro_scores=[]
fscore_macro_scores=[]
wordsembedding=[]
featureSelected={}
top10feature={}
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
data_corpus=documents.testo
print('--------')

print('print unique labels:', documents['cap_maj_master'].unique())
ordina=documents['cap_maj_master'].unique()
class_dic, labels=create_labels_dict(documents['cap_maj_master'].astype(int))
labels=np.asarray(labels)
#ordina=[1, 12, 18, 2, 9, 20, 15, 21, 7, 16, 3, 19, 5, 4, 17, 6, 10, 13, 14, 8, 23]

#print("Shape of labels: %d" %  labels.shape)
target_names = ["classe {}".format(x) for x in class_dic.keys()]
#print("print target_names")
#print(target_names)

target_dict={}
for i,classi in enumerate(target_names):
        target_dict[classi]=i

#print('dizionario delle classi:')
#print(target_dict)        

#ordina=[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,23]
macroTopic=['Domestic Microeconomic Issues', 'Civil Right, Minority Issues, and Civil Liberties' , 'Health', 'Agriculture', 
            'Labour and Employment', 'Education', 'Environment', 'Energy', 'Immigration', 'Transportation', 'Low and Crime', 'Welfare',
            'C. Development and Housing Issue', 'Banking, Finance, and Domestic Commerce', 'Defence', 'Space, Science, Technology, and Communications',
            'Foreign Trade', 'International Affairs', 'Government Operations', 'Public Lands and Water Management', 'Cultural Policy Issues']  
 
NCLASSI=int(len(class_dic))
print('Numero di classi del problema: ', NCLASSI)

if EMBEDDING=='WORD2VEC_CNN':
    EMBEDDING_DIR = os.path.join(BASE_DIR, "embedding/word2vec_300_skipgram.txt")
    print('Carico Pre-trained Word Embedding word2vec.')
    raw_embedding = load_embedding_text(EMBEDDING_DIR)
if EMBEDDING=='GLOVE_CNN':
    EMBEDDING_DIR =  os.path.join(BASE_DIR, "embedding/glove_wiki_300ita.txt")
    print('Carico Pre-trained Word Embedding Glove.')
    raw_embedding= load_embedding_text(EMBEDDING_DIR)
if EMBEDDING=='FASTTEXT_CNN':
    EMBEDDING_DIR = os.path.join(BASE_DIR, "embedding/wiki.it.vec")
    print('Carico Pre-trained Word Embedding FastText.')
    raw_embedding= load_embedding_text(EMBEDDING_DIR)

############### START STRATIFIED CROSS VALIDATION WITH REPETITION ########################
cm_tot=0
k=0
time_start = time.time()
skf = RepeatedStratifiedKFold(n_splits=KNUM, n_repeats=2, random_state=random_state)
for train_index, test_index in skf.split(data_corpus, labels):    
    k+=1
    #if k>2:  continue
    print('--------INIZIO --- CICLO {0} ---------------' .format(k))
    X_train, X_test = data_corpus.iloc[train_index].values, data_corpus.iloc[test_index].values
    y_train, y_test = labels[train_index], labels[test_index]
    #y_train, y_test = target.iloc[train_index].values, target.iloc[test_index].values
    print('Shape of data train and data test:', X_train.shape,X_test.shape)
    print('Shape of label train and label test:', y_train.shape,y_test.shape)
    print("-------") 
    
    ### SET MODELs, INPUT SHAPE and OUTPUT DIRECTORY 
    if EMBEDDING=='BOW_SVM' or EMBEDDING=='BOW_{}_SVM'.format(max_words):
        model =svm.LinearSVC(C=10000,random_state=random_state)
        if FEATURE_SELECTION == 'True':
            EMBEDDING='BOW_{}_SVM'.format(max_words)
            OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
            # create fold path for results save
            foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
            if not os.path.exists(foldpath):
                os.makedirs(foldpath)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=stop_words)
            X_train = cvectorizer.fit_transform(raw_documents=X_train)
            X_test = cvectorizer.transform(X_test)           
            print("Shape of data train and data test after vectorization: ", X_train.shape)
            print("Shape of label train and label test after vectorization: ", X_test.shape)    
            X_train, X_test, feature_names = featureSelect(X_train,y_train,X_test,cvectorizer)
            print("Shape of data train and data test after features selection: ", X_train.shape)
            print("Shape of label train and label test after feature selection: ", X_test.shape) 
                                    
        else:
            OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
            # create fold path for results save
            foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
            if not os.path.exists(foldpath):
                os.makedirs(foldpath)
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=stop_words)
            X_train = vectorizer.fit_transform(raw_documents=X_train)
            X_test = vectorizer.transform(raw_documents=X_test)
            print("Shape of data train and data test after tfidf vectorization: ", X_train.shape)
            print("Shape of label train and label test after tfidf vectorization: ", X_test.shape)    
            
            with open(os.path.join(foldpath,"vectorizer_fold"+str(k)+"_"+EMBEDDING+".pickle"), 'wb') as handle:
                pickle.dump(vectorizer,handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    elif EMBEDDING=='BOW_CNB' or EMBEDDING=='BOW_{}_CNB'.format(max_words):
        model = ComplementNB(alpha=0.2)
        if FEATURE_SELECTION == 'True':
            EMBEDDING='BOW_{}_CNB'.format(max_words)
            OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
            # create fold path for results save
            foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
            if not os.path.exists(foldpath):
                os.makedirs(foldpath)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=stop_words)
            X_train = cvectorizer.fit_transform(raw_documents=X_train)
            X_test = cvectorizer.transform(X_test)
            print("Shape of data train and data test after vectorization: ", X_train.shape)
            print("Shape of label train and label test after vectorization: ", X_test.shape)    
            X_train, X_test, feature_names = featureSelect(X_train,y_train,X_test,cvectorizer)
            print("Shape of data train and data test after features selection: ", X_train.shape)
            print("Shape of label train and label test after feature selection: ", X_test.shape)  
        else:
            OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
            # create fold path for results save
            foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
            if not os.path.exists(foldpath):
                os.makedirs(foldpath)
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=stop_words)
            X_train = vectorizer.fit_transform(raw_documents=X_train)
            X_test = vectorizer.transform(raw_documents=X_test)
            print("Shape of data train and data test after tfidf vectorization: ", X_train.shape)
            print("Shape of label train and label test after tfidf vectorization: ", X_test.shape)    
            
            with open(os.path.join(foldpath,"vectorizer_fold"+str(k)+"_"+EMBEDDING+".pickle"), 'wb') as handle:
                pickle.dump(vectorizer,handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    elif EMBEDDING=='BOW_PAC' or EMBEDDING=='BOW_{}_PAC'.format(max_words):
        model = PassiveAggressiveClassifier(C=1.0, max_iter=1000, early_stopping=True, validation_fraction=VALIDATION_SPLIT, n_iter_no_change=4, random_state=random_state, tol=1e-3)
        #model = PassiveAggressiveClassifier(C=0.001, max_iter=1000, random_state=random_state)
        if FEATURE_SELECTION == 'True':
            EMBEDDING='BOW_{}_PAC'.format(max_words)
            OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
            # create fold path for results save
            foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
            if not os.path.exists(foldpath):
                os.makedirs(foldpath)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=stop_words)
            X_train = cvectorizer.fit_transform(raw_documents=X_train)
            X_test = cvectorizer.transform(X_test)
            print("Shape of data train and data test after vectorization: ", X_train.shape)
            print("Shape of label train and label test after vectorization: ", X_test.shape)    
            X_train, X_test, feature_names = featureSelect(X_train,y_train,X_test,cvectorizer)
            print("Shape of data train and data test after features selection: ", X_train.shape)
            print("Shape of label train and label test after feature selection: ", X_test.shape)   
        else:
            OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
            # create fold path for results save
            foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
            if not os.path.exists(foldpath):
                os.makedirs(foldpath)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH TF-IDF INDEX    
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=stop_words)
            X_train = vectorizer.fit_transform(raw_documents=X_train)
            X_test = vectorizer.transform(raw_documents=X_test)
            print("Shape of data train and data test after tfidf vectorization: ", X_train.shape)
            print("Shape of label train and label test after tfidf vectorization: ", X_test.shape)    
            
            with open(os.path.join(foldpath,"vectorizer_fold"+str(k)+"_"+EMBEDDING+".pickle"), 'wb') as handle:
                pickle.dump(vectorizer,handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    elif EMBEDDING=='BOW_MLP' or EMBEDDING=='BOW_{}_MLP'.format(max_words):
        model=create_sklearn_modelMLP()
        if FEATURE_SELECTION == 'True':
            EMBEDDING='BOW_{}_MLP'.format(max_words)
            OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
            # create fold path for results save
            foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
            if not os.path.exists(foldpath):
                os.makedirs(foldpath)
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH FEATURE SELECTION IG                                
            cvectorizer = CountVectorizer(lowercase=True, analyzer='word', encoding='utf-8', stop_words=stop_words)
            X_train = cvectorizer.fit_transform(raw_documents=X_train)
            X_test = cvectorizer.transform(X_test)
            print("Shape of data train and data test after vectorization: ", X_train.shape)
            print("Shape of label train and label test after vectorization: ", X_test.shape)    
            X_train, X_test, feature_names = featureSelect(X_train,y_train,X_test,cvectorizer)
            print("Shape of data train and data test after features selection: ", X_train.shape)
            print("Shape of label train and label test after feature selection: ", X_test.shape)  
                      
        else:
            OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)
            # create fold path for results save
            foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
            if not os.path.exists(foldpath):
                os.makedirs(foldpath)    
            
            # BUILD BAG OF WORDS TEXT RAPPRESENTATION WITH TF-IDF INDEX                    
            vectorizer = TfidfVectorizer(lowercase=True, analyzer = 'word',  decode_error='replace' , encoding='utf-8', stop_words=stop_words) 
            X_train = vectorizer.fit_transform(raw_documents=X_train)
            X_test = vectorizer.transform(raw_documents=X_test)
            print("Shape of data train and data test after tfidf vectorization: ", X_train.shape)
            print("Shape of label train and label test after tfidf vectorization: ", X_test.shape)   
            with open(os.path.join(foldpath,"vectorizer_fold"+str(k)+"_"+EMBEDDING+".pickle"), 'wb') as handle:
                pickle.dump(vectorizer,handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    elif EMBEDDING=='WORD2VEC_CNN' or EMBEDDING=='GLOVE_CNN' or EMBEDDING=='FASTTEXT_CNN':

        OUTPUT_DIR=os.path.join(BASE_DIR, EMBEDDING)         
        # create fold path for results save
        foldpath=os.path.join(OUTPUT_DIR, "fold" + str(k))
        if not os.path.exists(foldpath):
            os.makedirs(foldpath)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)
        sequences_train=tokenizer.texts_to_sequences(X_train)
        sequences_valid=tokenizer.texts_to_sequences(X_test)
        vocab_size = len(tokenizer.word_index)+1
        print('Found %s unique tokens.' % vocab_size)
        # pad sequences
        X_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        X_test = pad_sequences(sequences_valid,maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        y_train = to_categorical(y_train, NCLASSI)
        y_test = to_categorical(y_test, NCLASSI)
        #n_classes=y_train.shape[1]
    
        print('Shape of X train and X validation tensor:', X_train.shape,X_test.shape)
        print('Shape of label train and validation tensor:', y_train.shape,y_test.shape)
        print("-------")

        # set embedding weight matrix
        embedding_matrix=setEmbeddingMatrix2(raw_embedding,tokenizer.word_index)
        embedding_layer=Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    
        #embedding_layer=Embedding(vocab_size, EMBEDDING_DIM,  input_length=MAX_SEQUENCE_LENGTH, trainable=True)
        #save tokenizer as vocabulary
        with open(os.path.join(foldpath,"tokenizer_fold"+str(k)+"_"+EMBEDDING+".pickle"), 'wb') as handle:
            pickle.dump(tokenizer,handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        # Create callbacks
        name_weights = os.path.join(foldpath, "model_fold" + str(k) + "_"+EMBEDDING+".hdf5")
        checkpointer = ModelCheckpoint(filepath=name_weights,  verbose=2, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
        callbacks_list = [early_stopping,checkpointer]
    
        ### CREATE MODEL
        model=create_model()


    if EMBEDDING in DEEPMODEL:
        ### FIT MODEL
        #history=model.fit(X_train, y_train, batch_size=BATCH, validation_data=(X_test, y_test), callbacks=callbacks_list, epochs=EPOCHS, verbose=VERBOSE_FIT)
        history=model.fit(X_train, y_train, batch_size=BATCH, validation_split=VALIDATION_SPLIT, callbacks=callbacks_list, epochs=EPOCHS, verbose=VERBOSE_FIT)
        print("Validation Loss: %.2f%%  (+/- %.2f%%)" % (np.mean(history.history['val_loss']), np.std(history.history['val_loss'])))
        #print("Validation Accuracy: %.2f%%  (+/- %.2f%%)" % (np.mean(history.history['accuracy']), np.std(history.history['accuracy'])))
        print('----------')
        ### PREDICTION
        y_pred=model.predict(X_test)
        y_test, y_pred = np.argmax(y_test, axis=1) , np.argmax(y_pred,axis=1)
    else:
        ### FIT MODEL
        model.fit(X_train, y_train)
        ### PREDICTION
        y_pred=model.predict(X_test)
        # SAVE MODEL
        with open(os.path.join(foldpath, "model_fold" + str(k) + "_"+EMBEDDING+".pickle"), 'wb') as handle:
            pickle.dump(model,handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    if print_top10=='True' and FEATURE_SELECTION=='True':
        top10_features_for_classes(feature_names)

    ### EVALUATION
    score=accuracy_score(y_test, y_pred) * 100
    print(" Ciclo %d Accuracy: %.2f%%" % (k, score))
    cvscores.append(score)
    
    # COMPUTE CONFUSION MATRIX  
    cm=confusion_matrix(y_test, y_pred)
    cm_tot=cm_tot + cm
    #print(classification_report(y_test, y_pred)) 

    # COMPUTE PRECISION, RECALL E F1 SCORE METRICS    
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    prec_macro,recall_macro,fscore_macro,supp_macro = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    prec_macro_scores.append(prec_macro*100)
    recall_macro_scores.append(recall_macro*100)
    fscore_macro_scores.append(fscore_macro*100)
    print('precision macro: ' , prec_macro)
    print('recall macro: ' , recall_macro)
    print('F1score macro: ' , fscore_macro)
    # SAVE ALL CALSSES METRICS FOR EVERY FOLD IN A DATAFRAME 
    df=pd.DataFrame()
    i=range(NCLASSI)
    df=pd.DataFrame({'CLASSE' : pd.Series(target_names, index=i) , 'PRECISION' : pd.Series(precision, index=i).round(2),'RECALL' : pd.Series(recall, index = i).round(2), 'F1SCORE' : pd.Series(fscore, index = i).round(2)}) # 'AUC' : pd.Series(roc_auc, index = i).round(2)})
    results_tfidf=results_tfidf.append(df, ignore_index=True, sort=False)
    print('---------FINE ------ CICLO {0} ---------------' .format(k))

time_elapsed = (time.time() - time_start)
time_elapsed=time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
    
foldpath=os.path.join(OUTPUT_DIR, "totale")
if not os.path.exists(foldpath):
    os.makedirs(foldpath)
    
# Compute confusion matrix
if PLOT_CM=='True':
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    cm_tot=np.rint(cm_tot/(k*2))
    np.set_printoptions(precision=2)
    #Plot non-normalized confusion matrix
    plt.figure(6)
    plot_confusion_matrix(cm_tot.astype(int), classes=ordina,  title='Confusion matrix')    

## RISULTATI OTTENUTI PER OGNI CLASSE IN CROSS VALIDATION (MEDIA E DEVIAZIONE STANDARD)
results_tfidf['CLASSE']=results_tfidf.CLASSE.astype(str)
results_tfidf['PRECISION']=results_tfidf.PRECISION.astype(float)
results_tfidf['RECALL']=results_tfidf.RECALL.astype(float)
results_tfidf['F1SCORE']=results_tfidf.F1SCORE.astype(float)
#results_tfidf['AUC']=results_tfidf.AUC.astype(float)
results_tfidf.to_csv(os.path.join(foldpath,"{}_folds_classes_score.csv".format(EMBEDDING)), encoding='utf-8', sep=SEP)
df1_mean=results_tfidf.groupby('CLASSE', sort=False).mean().round(2)
df2_std=results_tfidf.groupby('CLASSE', sort=False).std().round(2)
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
results.set_index('TOPICS')
#idx=0
#results=results.insert(loc=idx, column='TOPICS', value=macroTopic)
print(results)
results.to_csv(os.path.join(foldpath,"{}_cv_classes_scores.csv".format(EMBEDDING)), encoding='utf-8', sep=SEP)

## RISULATI PER FOLDS IN CROSS VALIDATION (MEDIANDO TUTTE LE CLASSI IN OGNI CICLO DI ELABORAZIONE)
df_accuratezza_scores=pd.DataFrame()
ind=range(len(cvscores))
df_accuratezza_scores= pd.DataFrame({'Accuracy' : pd.Series(cvscores, index = ind).round(1), 
                                     'Precision': pd.Series(prec_macro_scores, index=ind).round(1), 
                                     'Recall': pd.Series(recall_macro_scores, index=ind).round(1), 
                                     'F1score': pd.Series(fscore_macro_scores, index=ind).round(1)})
df_accuratezza_scores.to_csv(os.path.join(foldpath,"{}_cv_folds.csv".format(EMBEDDING)), encoding='utf-8', sep=SEP)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

## RISULTATI TOTALE (MEDIANDO TUTTI I FOLDS)
df_totale_scores=pd.DataFrame([["{}".format(EMBEDDING), np.mean(cvscores).round(1),  np.std(cvscores).round(1), np.max(cvscores).round(1), 
                                                            np.mean(prec_macro_scores).round(1), np.std(prec_macro_scores).round(1), np.max(prec_macro_scores).round(1), 
                                                            np.mean(recall_macro_scores).round(1), np.std(recall_macro_scores).round(1), np.max(recall_macro_scores).round(1),
                                                            np.mean(fscore_macro_scores).round(1), np.std(fscore_macro_scores).round(1), np.max(fscore_macro_scores).round(1)]], 
                                                            columns=['Modello','accuracy_mean','accuracy_std','accuracy_max','precision_mean','precision_std','precision_max',
                                                                     'recall_mean','recall_std','recall_max','f1score_mean','f1score_std','f1score_max'])
df_totale_scores.to_csv(os.path.join(foldpath,"{}_cv_total.csv".format(EMBEDDING)), encoding='utf-8', sep=SEP)
## BEST TOP 10 FEATURES SCORE FOR FOLD
features_scores=pd.DataFrame.from_dict(featureSelected)
features_scores.to_csv(os.path.join(foldpath,"{}_top10features_score.csv".format(EMBEDDING)), encoding='utf-8', sep=SEP)
## TIME PROCESSIS
f = open(os.path.join(foldpath, "timelapsed.txt"), "w")
f.write("Time Elapsed: {0}".format(time_elapsed))
f.close()    

exit(0)