'''
Created on 07 lug 2020

@author: Utente
'''


## SET GLOBAL PARAMETERS
UPLOAD_FOLDER = '/tmp/'
DOWNLOAD_FOLDER = 'tmp'
ALLOWED_EXTENSIONS = set(['csv', 'xls', 'xlsx'])
### API CLASSIFIER
API_REST_MLP = "http://localhost:8085/mlp"
API_REST_MLPFS = "http://localhost:8085/mlpfs"
API_REST_SVM = "http://localhost:8085/svm"
API_REST_SVMFS = "http://localhost:8085/svmfs"
API_REST_CNB = "http://localhost:8085/cnb"
API_REST_CNBFS = "http://localhost:8085/cnbfs"
API_REST_PAC = "http://localhost:8085/pac"
API_REST_PACFS = "http://localhost:8085/pacfs"
API_REST_DL = "http://localhost:8085/deep"
### API TRAINER
API_REST_ML_TRAIN = "http://localhost:8085/mltrain"
#API_REST_SVM_TRAIN = "http://localhost:8085/svm"
#API_REST_CNB_TRAIN = "http://localhost:8085/cnb"
#API_REST_PAC_TRAIN = "http://localhost:8085/pac"
#API_REST_DL_TRAIN = "http://localhost:8085/deep"
###
#KERAS_REST_API_MLP_TRAIN = "http://localhost:8086/mlptrain"
#KERAS_REST_API_SVM_TRAIN = "http://localhost:8086/svmtrain"
#KERAS_REST_API_CNN_TRAIN = "http://localhost:8086/mlptrain"

# initialize constants used for server queuing
MLP_QUEUE = "mlp_queue"
SVM_QUEUE = "svm_queue"
CNB_QUEUE = "cnb_queue"
PAC_QUEUE = "pac_queue"
MLPFS_QUEUE = "mlp_fs_queue"
SVMFS_QUEUE = "svm_fs_queue"
CNBFS_QUEUE = "cnb_fs_queue"
PACFS_QUEUE = "pac_fs_queue"
DEEP_QUEUE = "deep_queue"
ML_TRAIN_QUEUE="ml_train_queue"
#CNN_QUEUE= "ccn_queue"
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25
REDISDB='127.0.0.1'

macroTopic=['Domestic Microeconomic Issues', 'Civil Right, Minority Issues, and Civil Liberties' , 'Health', 'Agriculture', 
            'Labour and Employment', 'Education', 'Environment', 'Energy', 'Immigration', 'Transportation', 'Low and Crime', 'Welfare',
            'C. Development and Housing Issue', 'Banking, Finance, and Domestic Commerce', 'Defence', 'Space, Science, Technology, and Communications',
            'Foreign Trade', 'International Affairs', 'Government Operations', 'Public Lands and Water Management', 'Cultural Policy Issues']
