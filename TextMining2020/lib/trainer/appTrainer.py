'''
Created on 07 lug 2020

@author: Utente
'''

import os, json
import pandas as pd
pd.set_option('display.max_colwidth',100)
import requests
from flask import Blueprint, render_template, flash, request, redirect, url_for, current_app
from config.websetting import API_REST_ML_TRAIN, UPLOAD_FOLDER, macroTopic

appTrainer = Blueprint('appTrainer', __name__)


@appTrainer.route('/trainer', methods=['GET', 'POST'])
def trainer():
    if request.method == 'POST':
        # get form parameters
        filepath=request.form.get('filepath')
        filtro=request.form.get('vocab')
        analisi=request.form.get('stratifiedCV') 
        kfold=request.form.get('kfold')
        modello=request.form.get('trainer')
        current_app.logger.debug("inserito file: %s", filepath)
        current_app.logger.debug("scelto filtro: %s", filtro)
        current_app.logger.debug("scelto analisi: %s", analisi)
        current_app.logger.debug("scelto numero fold: %s" , kfold)
        current_app.logger.debug("scelto modello: %s", modello)
        list_doc=[]
        list_label=[]
        if not  filepath:
            flash('Attenzione non sono stati caricati dati da classificare' , 'warning')
            current_app.logger.debug('Attenzione non sono stati caricati dati da addestrare')
            return redirect(url_for('train'))
        current_app.logger.debug('Presente file per addestramento modello')
        filepath=os.path.join(UPLOAD_FOLDER, filepath)
        if filepath.lower().endswith(('.xls', '.xlsx')):
            #xl = pd.ExcelFile(filepath, encoding='utf-8')
            xl = pd.ExcelFile(filepath)
            documents=xl.parse(xl.sheet_names[0])
            #documents = pd.read_excel(filepath, encoding='utf-8')
        else:
            documents = pd.read_csv(filepath, encoding='utf-8')
        if os.path.exists(filepath):
            os.remove(filepath)
        list_doc=documents['testo'].tolist()
        list_label=documents['cap_maj_master'].tolist()
        #set dictionary input for json serialization           
        doc={"testo": list_doc,'cap_maj_master': list_label, 'filtro': filtro, 'analisi': analisi, 'kfold': kfold, 'modello': modello}
        payload= json.dumps(doc, indent=4)
        current_app.logger.debug('Generato json')
        #app.logger.debug(payload)
        # send request to API REST endpoint
        current_app.logger.debug('Scelto modello da addestrare: %s', modello)
        r = requests.post(API_REST_ML_TRAIN, json=payload).json()
        
        #if modello == 'MLP':
        #    current_app.logger.debug('Scelto modello da addestrare: %s', modello)
        #    r = requests.post(KERAS_REST_API_MLP_TRAIN, json=payload).json()
        #elif modello == 'SVM':
        #    current_app.logger.debug('Scelto modello da addestrare: %s', modello)
        #    r = requests.post(KERAS_REST_API_SVM_TRAIN, json=payload).json()
        #else:
        #    current_app.logger.debug('Scelto modello da addestrare: %s', modello)
        #    r = requests.post(KERAS_REST_API_CNN_TRAIN, json=payload).json()
        
        # ensure the request was sucessful
        if r["success"]:
            #app.logger.debug(r)
            #return  render_template('trainer.html')
            if  analisi:
                current_app.logger.debug('Analisi kfold cross validation abilitata')
                i=range(int(kfold)*2)
                a=[x+1 for x in i]
                df=pd.DataFrame({'Fold': pd.Series(a, index=i),'Accuratezza' : pd.Series(r['trained']['Accuratezza'], index = i), 'Precisione': pd.Series(r['trained']['Precisione'], index = i), 
                                 'Recall': pd.Series(r['trained']['Recall'], index = i), 'F1score' : pd.Series(r['trained']['F1score'], index = i)})
                mbest=df['F1score'].idxmax(axis=0, skipna=True)
                current_app.logger.debug('Modello migliore %d', mbest)
                jsondf=json.loads(df.to_json(orient='records'))
                k=range(len(r['trained']['Classe']))
                df=pd.DataFrame({'Classe': pd.Series(r['trained']['Classe'], index=k),'Precisione': pd.Series(r['trained']['PPV'], index=k), 'Recall': pd.Series(r['trained']['TPR'], index=k), 
                                 'F1score': pd.Series(r['trained']['Fmeasure'], index=k)})
                jsonclass={'Classe': r['trained']['Classe'], 'Precisione': r['trained']['PPV'], 'Recall': r['trained']['TPR'], 'F1score': r['trained']['Fmeasure']}
                #df['Classe']=df.Classe.astype(int)
                #df=df.sort_values(by=['Classe'] )
                
                df = df.assign(Descrizione=pd.Series(macroTopic).values)
                jsonclass=json.loads(df.to_json(orient='records'))
                jsondtot={'best_score': int(mbest)+1, 'modello' : r['trained']['modello'],'Accuracy mean': r['trained']['Accuracy mean'], 'Accuracy std' : r['trained']['Accuracy std'], 'Accuracy max': r['trained']['Accuracy max'], 
                          'Precisione mean': r['trained']['Precision mean'], 'Precision std' : r['trained']['Precision std'], 'Precision max': r['trained']['Precision max'], 
                          'Recall mean': r['trained']['Recall mean'],'Recall std' : r['trained']['Recall std'], 'Recall max': r['trained']['Recall max'],
                          'F1score mean': r['trained']['F1score mean'],'F1score std' : r['trained']['F1score std'], 'F1score max': r['trained']['F1score max']}
                current_app.logger.debug(df)
                current_app.logger.debug(jsonclass)
                current_app.logger.debug(jsondtot)
                return  render_template('score.html', classes=jsonclass, results=jsondf, total=jsondtot)
            else:
                jsonTime={'modello' : r['trained']['modello'],'time_elapsed': r['trained']['time_elapsed'],'output_dir': r['trained']['output_dir']}
                return  render_template('score2.html', notResults=jsonTime)
                
                #k=range(len(r['trained']['Classe']))
                #df=pd.DataFrame({'Classe': pd.Series(r['trained']['Classe'], index=k),'Precisione': pd.Series(r['trained']['PPV'], index=k), 'Recall': pd.Series(r['trained']['TPR'], index=k), 'F1score': pd.Series(r['trained']['Fmeasure'], index=k)})
                #jsonclass={'Classe': r['trained']['Classe'], 'Precisione': r['trained']['PPV'], 'Recall': r['trained']['TPR'], 'F1score': r['trained']['Fmeasure']}
                #df['Classe']=df.Classe.astype(int)
                #df=df.sort_values(by=['Classe'] )
                #df = df.assign(Descrizione=pd.Series(macroTopic).values)
                #jsonclass=json.loads(df.to_json(orient='records'))
                #jsondtot={'modello' : r['trained']['modello'],'Accuratezza media': r['trained']['Accuratezza media'], 'Accuratezza std' : r['trained']['Accuratezza std'], 'Accuratezza max': r['trained']['Accuratezza max'], 'Precisione media': r['trained']['Precisione media'], 'Recall media': r['trained']['Recall media'], 'F1score media': r['trained']['F1score media']}
                #app.logger.debug(jsonclass)
                #app.logger.debug(jsondtot)
                #return  render_template('score.html',  classes=jsonclass, total=jsondtot)
        # otherwise, the request failed
        else:
            current_app.logger.debug("Request failed")
            return redirect(url_for('train'))
    return redirect(url_for("train"))