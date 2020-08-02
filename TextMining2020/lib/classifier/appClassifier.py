'''
Created on Mar 10, 2019

@author: silvio
'''
import os, json, time
import pandas as pd
pd.set_option('display.max_colwidth',100)
import requests
from flask import Blueprint, render_template, flash, request, redirect, url_for, current_app
from config.websetting import *

def gen_codice():
    import uuid
    stringLength = 8
    randomString = uuid.uuid4().hex # get a random string in a UUID fromat
    randomString  = randomString.upper()[0:stringLength]
    return randomString

appClassifier = Blueprint('appClassifier', __name__)

@appClassifier.route('/classifier', methods=['GET', 'POST'])
def classifier():
    if request.method == 'POST':
        # get form parameters
        filepath=request.form.get('filepath')
        codice=request.form.get('codice')
        testo=request.form.get('testo')
        modello=request.form.get('classifier')
        current_app.logger.debug("File Selected: %s", filepath)
        current_app.logger.debug("Code entered: %s", codice)
        current_app.logger.debug("Text entered: %s", testo)
        current_app.logger.debug("Model entered: %s", modello)
        list_doc=[]
        list_cod=[]
        if not  filepath and not testo:
            flash('Warning: no data uploaded for labeling' , 'warning')
            current_app.logger.debug('Warning: no data uploaded for labeling')
            return redirect(url_for('main'))
        if not filepath:
            if not codice:
                codice=gen_codice()
            current_app.logger.debug('Only text data present')
            current_app.logger.debug('Generate unique document identifier: %s', codice)    
            filepath=None
            list_doc.append(testo)
            list_cod.append(codice)
            #doc={'codice': list_cod, "testo": list_doc}
        if not testo:
            testo=None
            current_app.logger.debug('Only documents file present')
            filepath=os.path.join(current_app.config['UPLOAD_FOLDER'], filepath)
            if filepath.lower().endswith(('.xls', '.xlsx')):
                #xl = pd.ExcelFile(filepath, encoding='utf-8')
                xl = pd.ExcelFile(filepath)
                documents=xl.parse(xl.sheet_names[0])
                #documents = pd.read_excel(filepath, encoding='utf-8')
            else:
                documents = pd.read_csv(filepath, encoding='utf-8')
            if os.path.exists(filepath):
                os.remove(filepath)
            list_cod=documents['id'].tolist()
            list_doc=documents['testo'].tolist()
        if filepath and testo:
            if not codice:
                codice=gen_codice()
            current_app.logger.debug('select both documents file and text to be labeled')
            current_app.logger.debug('Generate unique document identifier: %s', codice)
            filepath=os.path.join(current_app.config['UPLOAD_FOLDER'], filepath)
            # if file xlsx or xls has more sheet, take always the first sheet
            if filepath.lower().endswith(('.xls', '.xlsx')):
                #xl = pd.ExcelFile(filepath, encoding='utf-8')
                xl = pd.ExcelFile(filepath)
                documents=xl.parse(xl.sheet_names[0])
            else:
                documents = pd.read_csv(filepath, encoding='utf-8')
            if os.path.exists(filepath):
                os.remove(filepath)
            list_cod=documents['id'].tolist()
            list_doc=documents['testo'].tolist()
            list_doc.append(testo)
            list_cod.append(codice)
        #set dictionary input for json serialization   
        doc={'codice': list_cod, "testo": list_doc}
        payload= json.dumps(doc, indent=4)
        current_app.logger.debug('json created')
        # send request to API REST endpoint
        if modello == 'BOW_MLP':
            current_app.logger.debug('Selected model: %s', modello)
            r = requests.post(API_REST_MLP, json=payload).json()
        elif modello == 'BOW_20000_MLP':
            current_app.logger.debug('Selected model:  %s', modello)
            r = requests.post(API_REST_MLPFS, json=payload).json()  
        elif modello == 'BOW_SVM':
            current_app.logger.debug('Selected model:  %s', modello)
            r = requests.post(API_REST_SVM, json=payload).json()
        elif modello == 'BOW_20000_SVM':
            current_app.logger.debug('Selected model:  %s', modello)
            r = requests.post(API_REST_SVMFS, json=payload).json()
        elif modello == 'BOW_CNB':
            current_app.logger.debug('Selected model:  %s', modello)
            r = requests.post(API_REST_CNB, json=payload).json()
        elif modello == 'BOW_20000_CNB':
            current_app.logger.debug('Selected model:  %s', modello)
            r = requests.post(API_REST_CNBFS, json=payload).json()
        elif modello == 'BOW_PAC':
            current_app.logger.debug('Selected model:  %s', modello)
            r = requests.post(API_REST_PAC, json=payload).json()
        elif modello == 'BOW_20000_PAC':
            current_app.logger.debug('Selected model:  %s', modello)
            r = requests.post(API_REST_PACFS, json=payload).json()    
        else:
            current_app.logger.debug('Selected model:  %s', modello)
            r = requests.post(API_REST_DL, json=payload).json()
        # ensure the request was sucessful
        df=pd.DataFrame(columns=['id', 'text', 'label'])
        i=0
        if r["success"]:
            for (i, result) in enumerate(r["prediction"]):
                # put results in dataframe object 
                record="id:  {}  label: {}".format(result['did'], result['classe'])
                df.loc[i, 'id'] = result['did']
                df.loc[i,'text' ]= result['testo']
                df.loc[i,'label']= result['classe']
                current_app.logger.debug(record)  
            timestr = time.strftime("%Y%m%d-%H%M%S")
            df.to_excel('./tmp/results-{}.xls'.format(timestr), index=False,  encoding='utf-8')
            df.to_csv('./tmp/results-{}.csv'.format(timestr), index=False,  encoding='utf-8')
            # serialize the dataframe object in json and send to results template
            jsonfile=json.loads(df.to_json(orient='records'))
            return  render_template('results.html', results=jsonfile, linkxls='results-{}.xls'.format(timestr), linkcsv='results-{}.csv'.format(timestr))
            #return  render_template('classifier.html', results_df=df.to_html(table_id='idTable',  index=False)
        # otherwise, the request failed
        else:
            current_app.logger.debug("Request failed")
            return redirect(url_for('main'))
    return redirect(url_for("main"))


