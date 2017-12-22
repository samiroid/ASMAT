import argparse
from collections import defaultdict
import cPickle
from ipdb import set_trace
import itertools
import numpy as np
import os
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import f1_score, accuracy_score
import sys
sys.path.append("..")
from ASMAT.lib import helpers
import json 

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
			# print "[reading feature @ {}]".format(fname+feat_suffix)
			x = np.load(data_path+feat_suffix)
			if X is None: 
				X = x
			else:
				X = np.concatenate((X,x),axis=1)
	return X, Y

def hypertune(train, dev, features, obj, confs_path, resp_path=None):
	X_train, Y_train = get_features(train, features)
	X_dev,  Y_dev  = get_features(dev, features)	
	confs = json.load(open(confs_path, 'r'))
	params = confs.keys()
	choices = confs.values()
	combs = list(itertools.product(*choices))
	hyperparams =  [{k:v for k,v in zip(c,x)} for x, c in zip(combs, [params]*len(combs))]
	best_hp = None
	best_score = 0
	for hp in hyperparams:
		#initialize model with the hyperparameters				
		model = SGDClassifier(random_state=1234,**hp)
		model.fit(X_train,Y_train)
		Y_hat = model.predict(X_dev)
		score = obj(Y_dev, Y_hat)
		print "[hyperparameters: {} | score: {}]".format(repr(hp), score)
		if score > best_score:
			best_score = score
			best_hp = hp
	print ""
	print "[best conf: {} | score: {}]".format(repr(best_hp),best_score)
	return best_hp, best_score

		# #avgF1 = f1_score(Y_dev, Y_hat,average="macro") 		
		# acc = accuracy_score(Y_dev, Y_hat)				
		# fname = os.path.basename(test)		
		# results = {"acc":round(acc,3),
		# 		"avgF1":round(avgF1,3),			   
		# 		"features":"+".join(features),
		# 		"dataset":fname,
		# 		"run_id":run_id,
		# 		"train_size":len(X_train),
		# 		"test_size":len(X_dev),
		# 		"hyper":repr(hp)}
		# cols = ["dataset", "run_id", "hyper", "acc", "avgF1"]
		# helpers.print_results(results, columns=cols)
		# if res_path is not None:
		# 	cols = ["dataset", "run_id", "features", "hyper", "acc", "avgF1"]
		# 	helpers.save_results(results, res_path, columns=results.keys())

def run(train, test, run_id, features, hyperparameters={}, res_path=None):
	X_train, Y_train = get_features(train, features)
	X_test,  Y_test  = get_features(test, features)	
	#train and evalute model	
	#initialize model with the hyperparameters	
	model = SGDClassifier(random_state=1234,**hyperparameters)
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
			"test_size":len(X_test),
			"hyper":repr(hyperparameters)}
	cols = ["dataset", "run_id", "hyper", "acc", "avgF1"]
	helpers.print_results(results, columns=cols)
	if res_path is not None:
		cols = ["dataset", "run_id", "features", "hyper", "acc", "avgF1"]
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
	par.add_argument('-hyperparams', type=str, default="", help='path to a dictionary of hyperparameters')

	return par

if __name__=="__main__":	
	parser = get_parser()
	args = parser.parse_args()  	
	#open datasets
	#train	
	print "[features: {}]".format("+".join(args.features))
	if args.run_id is None: args.run_id = "+".join(args.features)		
	if args.hyperparams is not None: 
		assert args.dev is not None, "Need a dev set for hyperparameter search"
	#loop through cross-validation folds (if any)
	if args.cv is None:			
		if len(args.hyperparams) > 0:
			print "[tuning hyperparameters from @ {}]".format(args.hyperparams)
			best_hyper, _ = hypertune(args.train, args.dev, args.features, \
									accuracy_score, args.hyperparams)
		else:
			best_hyper = {}
		#run model with the best hyperparams
		run(args.train, args.test, args.run_id, args.features, best_hyper, args.res_path)
	else:
		assert args.cv > 2, "need at leat 2 folds for cross-validation"
		for cv_fold in xrange(1, args.cv+1):
			if len(args.hyperparams) > 0:
				print "[tuning hyperparameters from @ {}]".format(args.hyperparams)
				best_hyper, _ = hypertune(args.train, args.dev, args.features, \
										f1_score, args.hyperparams)
			else:
				best_hyper = {}
			#run model with the best hyperparams
			run(args.train+"_"+str(cv_fold), args.test+"_"+str(cv_fold), \
				args.run_id+"_"+str(cv_fold), args.features, best_hyper, args.res_path)
			



	
