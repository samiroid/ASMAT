import argparse
from collections import defaultdict
import cPickle
from ipdb import set_trace
import numpy as np
import os
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import f1_score, accuracy_score
import sys
sys.path.append("..")
from ASMAT.lib import helpers

def get_features(data_path, features_path, verbose=True):	
	with open(data_path) as fid:
		train_data = cPickle.load(fid)
		X = None
		Y = np.array(train_data[1])		
		#remove extension from filename
		data_path = os.path.splitext(data_path)[0]
		fname = os.path.basename(data_path)
		#get features
		for ft in args.features:			
			feat_suffix = "_"+ft+".npy" 
			if verbose:
				print "[reading feature @ {}]".format(fname+feat_suffix)
			x = np.load(data_path+feat_suffix)
			if X is None: 
				X = x
			else:
				X = np.concatenate((X,x),axis=1)
	return X, Y

def get_parser():
    parser = argparse.ArgumentParser(description="Document Classifier")
    parser.add_argument('-train', type=str, required=True, help='train data')
    parser.add_argument('-dev', type=str, help='dev data')
    parser.add_argument('-test', type=str, help='test data')
    parser.add_argument('-features', type=str, required=True, nargs='+',
                        help='features')
    parser.add_argument('-run_id', type=str, help='run id')
    parser.add_argument('-res_path', type=str, help='results file')
    parser.add_argument('-verbose', type=int, default=1, 
    					help='verbosity: 1 (yes), 0 (no)')
    parser.add_argument('-pos_label', type=str, default="positive",
                        help='label for the positive class')
    parser.add_argument('-neg_label', type=str, default="negative",
                        help='label for the negative class')
    
    return parser

if __name__=="__main__":	
	parser = get_parser()
	args = parser.parse_args()  	
	#open datasets
	#train
	verbose = args.verbose == 1
	X_train, Y_train = get_features(args.train, args.features, verbose)
	X_test,  Y_test  = get_features(args.test, args.features, verbose)

	#train and evalute model
	model = SGDClassifier(random_state=1234)
	model.fit(X_train,Y_train)
	Y_hat = model.predict(X_test)
	
	avgF1 = f1_score(Y_test, Y_hat,average="macro") 		
	acc = accuracy_score(Y_test, Y_hat)				
	fname = os.path.basename(args.test)	

	run_id = args.run_id
	if run_id is None: run_id = "+".join(args.features)
	results = {"acc":round(acc,3),
			   "avgF1":round(avgF1,3),			   
			   "features":repr(args.features),
			   ">":fname,
			   "run_id":run_id}

	helpers.print_results(results,columns=[">","run_id","acc","avgF1"])
	if args.res_path is not None:
		helpers.save_results(results, args.res_path, columns=["run_id","acc","avgF1"])
	




	
