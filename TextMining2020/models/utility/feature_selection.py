'''
Created on 11 lug 2020

@author: Utente
'''

import os
import pickle
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from operator import itemgetter

import main

print_top10='False'
score_func=mutual_info_classif
featureSelected=[]

# feature selection
def featureSelect(X_train,y_train,X_test,cvectorizer,modello,output,k=1):
    # prepare vectorizer for select feature task 
    vectorizer = SelectKBest(score_func=score_func,k=main.MAX_WORDS)
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
            
    with open(os.path.join(output,"vectorizer_fold"+str(k)+"_"+modello+".pickle"), 'wb') as handle:
        pickle.dump(vectorizer,handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    return X_train_best, X_test_best, feature_names

