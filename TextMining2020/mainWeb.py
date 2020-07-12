'''
Created on Mar 10, 2019

@author: silvio
'''
import os
import pandas as pd
pd.set_option('display.max_colwidth',100)
import logging
from flask import Flask, render_template, flash, request, redirect, current_app, send_from_directory
from werkzeug.utils import secure_filename
from flask.logging import default_handler

from config.websetting import ALLOWED_EXTENSIONS, UPLOAD_FOLDER, DOWNLOAD_FOLDER
from lib.classifier.appClassifier import appClassifier
from lib.trainer.appTrainer import appTrainer

class RequestFormatter(logging.Formatter):
    def format(self, record):
        record.url = request.url
        record.remote_addr = request.remote_addr
        return super().format(record)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#df=pd.DataFrame(columns=['id', 'testo', 'classe'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.secret_key = 'djehd64656grey66gdey6dge6dneudhf4h6478'
app.config['SESSION_TYPE'] = 'filesystem'
app.logger.setLevel(logging.DEBUG)

## REGISTER MODULES
app.register_blueprint(appClassifier)
app.register_blueprint(appTrainer)


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
    
    formatter = RequestFormatter(
    '[%(asctime)s] %(remote_addr)s requested %(url)s\n'
    '%(levelname)s in %(module)s: %(message)s'
    )
    default_handler.setFormatter(formatter)
    
    app.run(host= '0.0.0.0', debug=True)
