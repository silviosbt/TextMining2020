'''
Created on 19 lug 2020

@author: Utente
'''

import argparse
from script.runTraining import run_ml_training
from script.runCrossValidation import run_ml_cv


macroTopic=['Domestic Microeconomic Issues', 'Civil Right, Minority Issues, and Civil Liberties' , 'Health', 'Agriculture', 
            'Labour and Employment', 'Education', 'Environment', 'Energy', 'Immigration', 'Transportation', 'Low and Crime', 'Welfare',
            'C. Development and Housing Issue', 'Banking, Finance, and Domestic Commerce', 'Defence', 'Space, Science, Technology, and Communications',
            'Foreign Trade', 'International Affairs', 'Government Operations', 'Public Lands and Water Management', 'Cultural Policy Issues']  
 

def parser_arguments():
    # Construct the argument parser
    parser = argparse.ArgumentParser(usage="%(prog)s [OPTION] DATASET", description='supervised training procedure for CAP dataset')
    parser.add_argument('-m', '--model', required=True, type=str, help='available Models:[SVM, CNB, PAC, MLP, CNN]')
    parser.add_argument('-d', '--basedir', default='/opt/workdir', help='base directory for save output (default: /opt/workdir)')
    parser.add_argument('-e', '--embedding', default='Word2Vec', help='choose pre-trained words-embedding [Word2Vec,FastText] (default: Word2Vec)')
    parser.add_argument('-fs','--feature_selection', action="store_true", help='enable supervised feature selection')
    parser.add_argument('-mw', '--max_words', default='20000', type=int, help='used by feature selection option (default: 20000)')
    parser.add_argument('-cm', '--plot_cm', action="store_true", help='plot confusion matrix')
    parser.add_argument('-cv', '--cross_validation', action="store_true", help='2 x 10-fold cross validation')
    parser.add_argument('dataset', metavar='DATASET', nargs='+', help='Dataset file in .xlsx, .xls or .cvs format')
    #args = parser.parse_args()
    #if args.feature_selection and (args.max_words is None):
    #    parser.error("--max_words is requires")
    
    args = vars(parser.parse_args())
    #print("model is {}\n basedir is {}\n embedding is {}\n feature_selection is {}\n dataset is {}".format(args['model'], args['basedir'], args['embedding'], args['feature_selection'], args['dataset'][0]))
    return args

if __name__ == "__main__":
    args=parser_arguments()
    if args['cross_validation']:
        print('run cross validation')
        run_ml_cv(args,macroTopic)
    else:
        print('not run cross validation')
        run_ml_training(args)

exit(0)




