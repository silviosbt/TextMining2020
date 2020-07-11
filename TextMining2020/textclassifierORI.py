'''
Created on Mar 10, 2019

@author: silvio
'''
import os, json, time
import pandas as pd
pd.set_option('display.max_colwidth',100)
import requests
import logging
from flask import Flask, render_template,  flash, request, redirect, url_for, current_app, send_from_directory
from werkzeug.utils import secure_filename
from flask.logging import default_handler

class RequestFormatter(logging.Formatter):
    def format(self, record):
        record.url = request.url
        record.remote_addr = request.remote_addr
        return super().format(record)

formatter = RequestFormatter(
    '[%(asctime)s] %(remote_addr)s requested %(url)s\n'
    '%(levelname)s in %(module)s: %(message)s'
)
default_handler.setFormatter(formatter)


UPLOAD_FOLDER = '/tmp/'
DOWNLOAD_FOLDER = 'tmp'
ALLOWED_EXTENSIONS = set(['csv', 'xls', 'xlsx'])
API_REST_MLP = "http://localhost:8085/mlp"
API_REST_MLPFS = "http://localhost:8085/mlpfs"
API_REST_SVM = "http://localhost:8085/svm"
API_REST_SVMFS = "http://localhost:8085/svmfs"
API_REST_CNB = "http://localhost:8085/cnb"
API_REST_CNBFS = "http://localhost:8085/cnbfs"
API_REST_PAC = "http://localhost:8085/pac"
API_REST_PACFS = "http://localhost:8085/pacfs"
API_REST_DL = "http://localhost:8085/deep"
KERAS_REST_API_MLP_TRAIN = "http://localhost:8086/mlptrain"
KERAS_REST_API_SVM_TRAIN = "http://localhost:8086/svmtrain"
KERAS_REST_API_CNN_TRAIN = "http://localhost:8086/mlptrain"
#df=pd.DataFrame(columns=['id', 'testo', 'classe'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.secret_key = 'djehd64656grey66gdey6dge6dneudhf4h6478'
app.config['SESSION_TYPE'] = 'filesystem'
app.logger.setLevel(logging.DEBUG)
macroTopic=['Domestic Microeconomic Issues', 'Civil Right, Minority Issues, and Civil Liberties' , 'Health', 'Agriculture', 
            'Labour and Employment', 'Education', 'Environment', 'Energy', 'Immigration', 'Transportation', 'Low and Crime', 'Welfare',
            'C. Development and Housing Issue', 'Banking, Finance, and Domestic Commerce', 'Defence', 'Space, Science, Technology, and Communications',
            'Foreign Trade', 'International Affairs', 'Government Operations', 'Public Lands and Water Management', 'Cultural Policy Issues']

def gen_codice():
    import uuid
    stringLength = 8
    randomString = uuid.uuid4().hex # get a random string in a UUID fromat
    randomString  = randomString.upper()[0:stringLength]
    return randomString

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            app.logger.debug("nessun file inserito")
            flash("Nessun file inserito", category='info')
            return redirect("/")
        file = request.files['file']
        app.logger.debug("inserito il file: %s", file)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            app.logger.debug('Nessun file selezionato')
            flash("Nessun file selezionato", category='info')
            return redirect("/")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            app.logger.debug("file salvato: %s", filename)
            #flash("caricato file di input: {}".format(filename), category='info')
            return render_template('upload.html', filename=filename)
        else:
            flash("file di input: {} non valido, file validi .xlsx ,.xls e .csv ".format(file.filename), category='info')
    return  redirect("/")

@app.route('/upload2', methods=['GET', 'POST'])
def upload2_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            app.logger.debug("nessun file inserito")
            #flash("Nessun file inserito", category='info')
            return redirect("/train")
        file = request.files['file']
        app.logger.debug("inserito il file: %s", file)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            app.logger.debug('Nessun file selezionato')
            flash("Nessun file selezionato", category='info')
            return redirect("/train")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            app.logger.debug("file salvato: %s", filename)
            #flash("caricato file di input: {}".format(filename), category='info')
            return render_template('upload2.html', filename=filename)
        else:
            flash("file di input: {} non valido, file validi .xlsx ,.xls e .csv ".format(file.filename), category='info')
    return  redirect("/train")


@app.route('/classifier', methods=['GET', 'POST'])
def classifier():
    if request.method == 'POST':
        # get form parameters
        filepath=request.form.get('filepath')
        codice=request.form.get('codice')
        testo=request.form.get('testo')
        modello=request.form.get('classifier')
        app.logger.debug("inserito file: %s", filepath)
        app.logger.debug("inserito codice: %s", codice)
        app.logger.debug("inserito testo: %s", testo)
        app.logger.debug("inserito modello: %s", modello)
        list_doc=[]
        list_cod=[]
        if not  filepath and not testo:
            #flash('Attenzione non sono stati caricati dati da classificare' , 'warning')
            app.logger.debug('Attenzione non sono stati caricati dati da classificare')
            return redirect(url_for('main'))
        if not filepath:
            if not codice:
                codice=gen_codice()
            app.logger.debug('Presente solo testo da classificare')
            app.logger.debug('generato codice: %s', codice)    
            filepath=None
            list_doc.append(testo)
            list_cod.append(codice)
            #doc={'codice': list_cod, "testo": list_doc}
        if not testo:
            testo=None
            app.logger.debug('Presente solo file da classificare')
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filepath)
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
            app.logger.debug('Presente  sia file che testo da classificare')
            app.logger.debug('generato codice: %s', codice)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filepath)
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
        app.logger.debug('Generato json')
        # send request to API REST endpoint
        if modello == 'BOW_MLP':
            app.logger.debug('Scelto modello di classificatore: %s', modello)
            r = requests.post(API_REST_MLP, json=payload).json()
        elif modello == 'BOW_20000_MLP':
            app.logger.debug('Scelto modello di classificatore: %s', modello)
            r = requests.post(API_REST_MLPFS, json=payload).json()  
        elif modello == 'BOW_SVM':
            app.logger.debug('Scelto modello di classificatore: %s', modello)
            r = requests.post(API_REST_SVM, json=payload).json()
        elif modello == 'BOW_20000_SVM':
            app.logger.debug('Scelto modello di classificatore: %s', modello)
            r = requests.post(API_REST_SVMFS, json=payload).json()
        elif modello == 'BOW_CNB':
            app.logger.debug('Scelto modello di classificatore: %s', modello)
            r = requests.post(API_REST_CNB, json=payload).json()
        elif modello == 'BOW_20000_CNB':
            app.logger.debug('Scelto modello di classificatore: %s', modello)
            r = requests.post(API_REST_CNBFS, json=payload).json()
        elif modello == 'BOW_PAC':
            app.logger.debug('Scelto modello di classificatore: %s', modello)
            r = requests.post(API_REST_PAC, json=payload).json()
        elif modello == 'BOW_20000_PAC':
            app.logger.debug('Scelto modello di classificatore: %s', modello)
            r = requests.post(API_REST_PACFS, json=payload).json()    
        else:
            app.logger.debug('Scelto modello di classificatore: %s', modello)
            r = requests.post(API_REST_DL, json=payload).json()
        # ensure the request was sucessful
        df=pd.DataFrame(columns=['id', 'testo', 'classe'])
        i=0
        if r["success"]:
            for (i, result) in enumerate(r["prediction"]):
                # put results in dataframe object 
                record="id:  {}  classe: {}".format(result['did'], result['classe'])
                df.loc[i, 'id'] = result['did']
                df.loc[i,'testo' ]= result['testo']
                df.loc[i,'classe']= result['classe']
                app.logger.debug(record)  
            timestr = time.strftime("%Y%m%d-%H%M%S")
            df.to_excel('./tmp/results-{}.xls'.format(timestr), index=False,  encoding='utf-8')
            df.to_csv('./tmp/results-{}.csv'.format(timestr), index=False,  encoding='utf-8')
            # serialize the dataframe object in json and send to results template
            jsonfile=json.loads(df.to_json(orient='records'))
            return  render_template('results.html', results=jsonfile, linkxls='results-{}.xls'.format(timestr), linkcsv='results-{}.csv'.format(timestr))
            #return  render_template('classifier.html', results_df=df.to_html(table_id='idTable',  index=False)
        # otherwise, the request failed
        else:
            app.logger.debud("Request failed")
            return redirect(url_for('main'))
    return redirect(url_for("main"))


@app.route('/trainer', methods=['GET', 'POST'])
def trainer():
    if request.method == 'POST':
        # get form parameters
        filepath=request.form.get('filepath')
        vocab=request.form.get('vocab')
        analisi=request.form.get('stratifiedCV') 
        kfold=request.form.get('kfold')
        modello=request.form.get('trainer')
        app.logger.debug("inserito file: %s", filepath)
        app.logger.debug("scelto vocab: %s", vocab)
        app.logger.debug("scelto analisi: %s", analisi)
        app.logger.debug("scelto numero fold: %s" , kfold)
        app.logger.debug("scelto modello: %s", modello)
        list_doc=[]
        list_label=[]
        if not  filepath:
            flash('Attenzione non sono stati caricati dati da classificare' , 'warning')
            app.logger.debug('Attenzione non sono stati caricati dati da addestrare')
            return redirect(url_for('train'))
        app.logger.debug('Presente file per addestramento modello')
        filepath=os.path.join(app.config['UPLOAD_FOLDER'], filepath)
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
        doc={"testo": list_doc,'cap_maj_master': list_label, 'vocab': vocab, 'analisi': analisi, 'kfold': kfold, 'modello': modello}
        payload= json.dumps(doc, indent=4)
        app.logger.debug('Generato json')
        #app.logger.debug(payload)
        # send request to API REST endpoint
        if modello == 'MLP':
            app.logger.debug('Scelto modello da addestrare: %s', modello)
            r = requests.post(KERAS_REST_API_MLP_TRAIN, json=payload).json()
        elif modello == 'SVM':
            app.logger.debug('Scelto modello da addestrare: %s', modello)
            r = requests.post(KERAS_REST_API_SVM_TRAIN, json=payload).json()
        else:
            app.logger.debug('Scelto modello da addestrare: %s', modello)
            r = requests.post(KERAS_REST_API_CNN_TRAIN, json=payload).json()
        # ensure the request was sucessful
        if r["success"]:
            #app.logger.debug(r)
            #return  render_template('trainer.html')
            if  analisi:
                app.logger.debug('Analisi kfold cross validation abilitata')
                i=range(int(kfold))
                a=[x+1 for x in i]
                df=pd.DataFrame({'Fold': pd.Series(a, index=i),'Accuratezza' : pd.Series(r['trained']['Accuratezza'], index = i), 'Precisione': pd.Series(r['trained']['Precisione'], index = i), 'Recall': pd.Series(r['trained']['Recall'], index = i), 'F1score' : pd.Series(r['trained']['F1score'], index = i)})
                mbest=df['Accuratezza'].idxmax(axis=0, skipna=True)
                app.logger.debug('Modello migliore %d', mbest)
                jsondf=json.loads(df.to_json(orient='records'))
                k=range(len(r['trained']['Classe']))
                df=pd.DataFrame({'Classe': pd.Series(r['trained']['Classe'], index=k),'Precisione': pd.Series(r['trained']['PPV'], index=k), 'Recall': pd.Series(r['trained']['TPR'], index=k), 'F1score': pd.Series(r['trained']['Fmeasure'], index=k)})
                #jsonclass={'Classe': r['trained']['Classe'], 'Precisione': r['trained']['PPV'], 'Recall': r['trained']['TPR'], 'F1score': r['trained']['Fmeasure']}
                df['Classe']=df.Classe.astype(int)
                df=df.sort_values(by=['Classe'] )
                df = df.assign(Descrizione=pd.Series(macroTopic).values)
                jsonclass=json.loads(df.to_json(orient='records'))
                jsondtot={'best_score': int(mbest)+1, 'modello' : r['trained']['modello'],'Accuratezza media': r['trained']['Accuratezza media'], 'Accuratezza std' : r['trained']['Accuratezza std'], 'Accuratezza max': r['trained']['Accuratezza max'], 'Precisione media': r['trained']['Precisione media'], 'Recall media': r['trained']['Recall media'], 'F1score media': r['trained']['F1score media']}
                app.logger.debug(df)
                app.logger.debug(jsonclass)
                app.logger.debug(jsondtot)
                return  render_template('score.html', classes=jsonclass, results=jsondf, total=jsondtot)
            else:
                k=range(len(r['trained']['Classe']))
                df=pd.DataFrame({'Classe': pd.Series(r['trained']['Classe'], index=k),'Precisione': pd.Series(r['trained']['PPV'], index=k), 'Recall': pd.Series(r['trained']['TPR'], index=k), 'F1score': pd.Series(r['trained']['Fmeasure'], index=k)})
                #jsonclass={'Classe': r['trained']['Classe'], 'Precisione': r['trained']['PPV'], 'Recall': r['trained']['TPR'], 'F1score': r['trained']['Fmeasure']}
                df['Classe']=df.Classe.astype(int)
                df=df.sort_values(by=['Classe'] )
                df = df.assign(Descrizione=pd.Series(macroTopic).values)
                jsonclass=json.loads(df.to_json(orient='records'))
                jsondtot={'modello' : r['trained']['modello'],'Accuratezza media': r['trained']['Accuratezza media'], 'Accuratezza std' : r['trained']['Accuratezza std'], 'Accuratezza max': r['trained']['Accuratezza max'], 'Precisione media': r['trained']['Precisione media'], 'Recall media': r['trained']['Recall media'], 'F1score media': r['trained']['F1score media']}
                #app.logger.debug(jsonclass)
                #app.logger.debug(jsondtot)
                return  render_template('score.html',  classes=jsonclass, total=jsondtot)
        # otherwise, the request failed
        else:
            app.logger.debud("Request failed")
            return redirect(url_for('train'))
    return redirect(url_for("train"))

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/train')
def train():
    return render_template('trainer.html')

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    downloads = os.path.join(current_app.root_path, app.config['DOWNLOAD_FOLDER'])
    return send_from_directory(directory=downloads, filename=filename)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(host= '0.0.0.0', debug=True)
