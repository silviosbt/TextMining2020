'''
Created on 26 lug 2020

@author: Utente
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


EXT='png'
    
def plot_confusion_matrix(data,labels,foldpath):
    df_cm = pd.DataFrame(data, columns=labels, index = labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.0)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
    plotsave=os.path.join(foldpath,"confusion_matrix.{}".format(EXT) )
    plt.savefig(plotsave ,format=EXT) #dpi=1000, bbox_inches='tight')
    #plt.show()
    plt.close(6)    