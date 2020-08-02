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
        current_app.logger.debug("File selected: %s", filepath)
        current_app.logger.debug("Filtro entered: %s", filtro)
        current_app.logger.debug("Analysis entered: %s", analisi)
        current_app.logger.debug("Fold number entered: %s" , kfold)
        current_app.logger.debug("Model selected: %s", modello)
        list_doc=[]
        list_label=[]
        if not  filepath:
            flash('Warning: no data uploaded for training' , 'warning')
            current_app.logger.debug('Warning: no data uploaded for training')
            return redirect(url_for('train'))
        current_app.logger.debug('Uploaded file for training model')
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
        current_app.logger.debug('Start training job..')
        r = requests.post(API_REST_ML_TRAIN, json=payload).json()
        current_app.logger.debug('labeled data received..')

        # ensure the request was sucessful
        if r["success"]:
            #app.logger.debug(r)
            #return  render_template('trainer.html')
            if  analisi:
                current_app.logger.debug('Stratified kfold cross validation analysis enabled')
                i=range(int(kfold)*2)
                a=[x+1 for x in i]
                df=pd.DataFrame({'Fold': pd.Series(a, index=i),'Accuratezza' : pd.Series(r['trained']['Accuratezza'], index = i), 'Precisione': pd.Series(r['trained']['Precisione'], index = i), 
                                 'Recall': pd.Series(r['trained']['Recall'], index = i), 'F1score' : pd.Series(r['trained']['F1score'], index = i)})
                mbest=df['F1score'].idxmax(axis=0, skipna=True)
                current_app.logger.debug('Best Model %d', mbest)
                jsondf=json.loads(df.to_json(orient='records'))
                k=range(len(r['trained']['Classe']))
                df=pd.DataFrame({'Classe': pd.Series(r['trained']['Classe'], index=k),'Precisione': pd.Series(r['trained']['PPV'], index=k), 'Recall': pd.Series(r['trained']['TPR'], index=k), 
                                 'F1score': pd.Series(r['trained']['Fmeasure'], index=k)})
                jsonclass={'Classe': r['trained']['Classe'], 'Precisione': r['trained']['PPV'], 'Recall': r['trained']['TPR'], 'F1score': r['trained']['Fmeasure']}
                
                df = df.assign(Descrizione=pd.Series(macroTopic).values)
                jsonclass=json.loads(df.to_json(orient='records'))
                jsondtot={'best_score': int(mbest)+1, 'modello' : r['trained']['modello'],'Accuracy mean': r['trained']['Accuracy mean'], 'Accuracy std' : r['trained']['Accuracy std'], 'Accuracy max': r['trained']['Accuracy max'], 
                          'Precision mean': r['trained']['Precision mean'], 'Precision std' : r['trained']['Precision std'], 'Precision max': r['trained']['Precision max'], 
                          'Recall mean': r['trained']['Recall mean'],'Recall std' : r['trained']['Recall std'], 'Recall max': r['trained']['Recall max'],
                          'F1score mean': r['trained']['F1score mean'],'F1score std' : r['trained']['F1score std'], 'F1score max': r['trained']['F1score max']}
                current_app.logger.debug(df)
                current_app.logger.debug(jsonclass)
                current_app.logger.debug(jsondtot)
                return  render_template('score.html', classes=jsonclass, results=jsondf, total=jsondtot)
            else:
                jsonTime={'modello' : r['trained']['modello'],'time_elapsed': r['trained']['time_elapsed'],'output_dir': r['trained']['output_dir']}
                return  render_template('score2.html', notResults=jsonTime)

        # otherwise, the request failed
        else:
            current_app.logger.debug("Request failed")
            return redirect(url_for('train'))
    return redirect(url_for("train"))