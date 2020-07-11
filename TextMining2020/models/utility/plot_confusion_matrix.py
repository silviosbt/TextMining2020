'''
Created on 11 lug 2020

@author: Utente
'''
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

from models.utility.cmaps import cmaps 
cmaps=cmaps()
Blues=cmaps.Blues

EXT='png'

def plot_confusion_matrix(cmt, classes, output, normalize=False, title='Confusion matrix', cmap=Blues):
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
    plotsave=os.path.join(output,"confusion_matrix.{}".format(EXT) )
    plt.savefig(plotsave ,format=EXT, dpi=1000, bbox_inches='tight')    
    #plt.show()
    plt.close(6)