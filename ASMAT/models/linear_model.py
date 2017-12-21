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

def get_features(data_path, features_path):	
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
			print "[reading feature @ {}]".format(fname+feat_suffix)
			x = np.load(data_path+feat_suffix)
			if X is None: 
				X = x
			else:
				X = np.concatenate((X,x),axis=1)
	return X, Y

def run(train, test, run_id, features, res_path=None):
	X_train, Y_train = get_features(train, features)
	X_test,  Y_test  = get_features(test, features)

	#train and evalute model
	model = SGDClassifier(random_state=1234)
	model.fit(X_train,Y_train)
	Y_hat = model.predict(X_test)

	avgF1 = f1_score(Y_test, Y_hat,average="macro") 		
	acc = accuracy_score(Y_test, Y_hat)				
	fname = os.path.basename(test)		
	results = {"acc":round(acc,3),
			   "avgF1":round(avgF1,3),			   
			   "features":"+".join(features),
			   "dataset":fname,
			   "run_id":run_id,
			   "train_size":len(X_train),
			   "test_size":len(X_test)}
	cols = ["dataset", "run_id", "features", "acc", "avgF1","train_size","test_size"]
	helpers.print_results(results, columns=cols)
	if res_path is not None:
		helpers.save_results(results, res_path, columns=cols)
	
def get_parser():
	par = argparse.ArgumentParser(description="Document Classifier")
	par.add_argument('-train', type=str, required=True, help='train data')
	par.add_argument('-dev', type=str, help='dev data')
	par.add_argument('-test', type=str, required=True, help='test data')
	par.add_argument('-features', type=str, required=True, nargs='+', help='features')
	par.add_argument('-run_id', type=str, help='run id')
	par.add_argument('-res_path', type=str, help='results file')
	par.add_argument('-silent', action="store_true",help='silent')
	par.add_argument('-pos_label', type=str, default="positive", \
					help='label for the positive class')
	par.add_argument('-neg_label', type=str, default="negative", \
					help='label for the negative class')
	par.add_argument('-cv', type=int, help='crossfold')

	return par

if __name__=="__main__":	
	parser = get_parser()
	args = parser.parse_args()  	
	#open datasets
	#train	
	
	if args.run_id is None: args.run_id = "+".join(args.features)
	#loop through cross-validation folds (if any)
	if args.cv is None:
		run(args.train, args.test, args.features, args.run_id, args.res_path)
	else:
		assert args.cv > 2, "need at leat 2 folds for cross-validation"
		for cv_fold in xrange(1, args.cv+1):
			run(args.train+"_"+str(cv_fold), \
				args.test+"_"+str(cv_fold), \
				args.run_id+"_"+str(cv_fold), \
				args.features, \
				args.res_path)
			



	
