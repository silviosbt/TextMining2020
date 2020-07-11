'''
Created on 11 lug 2020

@author: Utente
'''

def create_labels_dict(items):
    classi=items.unique()
    dic={}
    for i,classi in enumerate(classi):
        dic[classi]=i
    labels=items.apply(lambda x:dic[x])
    return dic, labels