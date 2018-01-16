import argparse
import cPickle
from collections import defaultdict
from ipdb import set_trace
import pprint
import json
import math
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sys
sys.path.append("..")
from ASMAT.lib.data import read_dataset
from ASMAT.lib import helpers
from ASMAT.models.lexicon import LexiconSentiment

def hypertune(lex_path, test, label_map,  obj, hyperparams, res_path=None):
	dt = read_dataset(test, labels=label_map.keys())
	X = [x[1] for x in dt]
	Y = [label_map[x[0]] for x in dt]		
	best_hp = None
	best_score = 0
	for hp in hyperparams:
		#initialize model with the hyperparameters			
		model = LexiconSentiment(path=lex_path,**hp)	
		Y_hat = model.predict(X)					
		score = obj(Y, Y_hat)
		# print "[score: {} | hyperparameters: {}]".format(score, repr(hp))
		if score > best_score:
			best_score = score
			best_hp = hp
		results = {"score":round(score,3), "hyper":repr(hp)}
		if res_path is not None:
			helpers.save_results(results,res_path)
		helpers.print_results(results)
	print ""
	print "[best conf: {} | score: {}]".format(repr(best_hp),best_score)
	return best_hp, best_score

def main(lex_path, test, label_map, run_id, hyperparams={}, res_path=None):
	dt = read_dataset(test, labels=label_map.keys())
	X = [x[1] for x in dt]
	Y = [label_map[x[0]] for x in dt]
	print "[configs]"
	pprint.pprint(hyperparams)
	model = LexiconSentiment(path=lex_path,**hyperparams)	
	Y_hat = model.predict(X)		
	avgF1 = f1_score(Y, Y_hat,average="macro") 		
	acc = accuracy_score(Y, Y_hat)				
	run_id = args.run_id
	if run_id is None:
	    run_id = os.path.splitext(os.path.basename(args.lex))[0]
	
	results = {"acc":round(acc,3), 
			"avgF1":round(avgF1,3),	
			"model":args.lex, 
			"dataset":os.path.basename(test), 
			"run_id":run_id, 
			"hyper":repr(hyperparams)}
	cols = ["dataset", "run_id", "acc", "avgF1","hyper"]
	
	helpers.print_results(results, columns=cols)
	if res_path is not None:
		helpers.save_results(results, res_path, columns=cols)
	
	return results

def summary(fname,df):
	print "saving to ", fname
	df.to_csv(fname,index=False)
	with open(fname,"a") as fid:
		avg_score = round(df["score"].mean(),3)
		std_score = round(df["score"].std(),3)
		avg_std = round(df["std"].mean(),3)
		fid.write("\n\navg score:{}\nstd score:{}\navg std:{}\n".format(avg_score, 
																	std_score,
																	avg_std))

def debug(lex_path, test, label_map, conf, report_path):
	dt = read_dataset(test, labels=label_map.keys())
	X = [x[1] for x in dt]
	Y = [label_map[x[0]] for x in dt]
	model = LexiconSentiment(path=lex_path,**conf)	
	z = model.debug(X)		
	out = [[true_y] + y_hat for true_y, y_hat in zip(Y,z)]	
	df = pd.DataFrame(out,columns=["y", "y_hat", "score", "std", "word_scores"])
	df.sort_values("score",inplace=True,ascending=False)	
	if not os.path.exists(os.path.dirname(report_path)):
		os.makedirs(os.path.dirname(report_path))
	#all instances
	summary(os.path.join(report_path, "all.txt"),df)	
	#true positives
	true_positives = df[df["y"] == 1]
	summary(os.path.join(report_path,"positives.txt"), true_positives)	
	#true negatives
	true_negatives = df[df["y"] == -1]
	summary(os.path.join(report_path,"negatives.txt"),true_negatives)
	#mistakes
	mistakes = df[df["y"] != df["y_hat"]]	
	summary(os.path.join(report_path,"mistakes.txt"),mistakes)

def get_args():
	par = argparse.ArgumentParser(description="Lexicon Classifier")    
	#Basic Input
	par.add_argument('-lex', type=str, required=True, help='lexicon file')
	par.add_argument('-test_set', type=str, required=True, help='test file(s)')
	par.add_argument('-dev_set', type=str, help='dev file(s)')
	par.add_argument('-positive_threshold', type=float, default=0.5, \
						help='documents with scores above are considered positive')
	par.add_argument('-negative_threshold', type=float, help='documents with scores below are considered negative')    
	par.add_argument('-default_word_score', type=float, help='score for words not present in the lexicon')    
	par.add_argument('-model', type=str, choices=["sum","mean"],default="mean")
	par.add_argument('-norm_scores', action="store_true", help='normalize word scores to range ][-1,1]')	
	par.add_argument('-pos_label', type=str, default="positive", 
					help='label for the positive class')
	par.add_argument('-neg_label', type=str, default="negative",
						help='label for the negative class')
	par.add_argument('-neut_label', type=str, 
					 help='label for the neutral class')	
	par.add_argument('-res_path', type=str, help='results file')
	par.add_argument('-run_id', type=str, help='run id')
	par.add_argument('-out', type=str, help='out')
	par.add_argument('-cv', type=int, help='cv')	
	par.add_argument('-debug', type=str, help='debug mode: path to dump the reports')	
	par.add_argument('-confs_path', type=str, help='path to configuration')
	par.add_argument('-hyperparams_path', type=str, default="",help='path to configuration')
	args = par.parse_args()  	
	return args

if __name__=="__main__":	
	args = get_args()
	label_map = {args.pos_label: 1, args.neg_label: -1}
	if args.neut_label is not None: label_map[args.neut_label] = 0	

	default_conf = {"default_word_score":args.default_word_score,
					"positive_threshold":args.positive_threshold, 
					"agg":args.model,
					"norm_scores":args.norm_scores,
					"default_class":1}
	if args.confs_path is not None:
		confs = json.load(open(args.confs_path, 'r'))
		#add configurations that are not specified with the default values
		default_conf = dict(default_conf.items() + confs.items())

	hyperparams_grid = []
	if len(args.hyperparams_path) > 0:				
		hyperparams_grid = helpers.get_hyperparams(args.confs_path, default_conf)
		print "[tuning hyperparameters from @ {}]".format(args.hyperparams_path)
		if args.res_path is not None:            
			fname, _ = os.path.splitext(args.res_path)            
			hyper_results_path = fname+"_"+os.path.basename(args.test_set)+"_hyper.txt"
		else:
			hyper_results_path = None
		scorer = lambda y_true,y_hat: f1_score(y_true, y_hat,average="macro") 	
	
	# RUN	
	if args.cv is None:			
		if len(hyperparams_grid) > 0:	
			assert args.dev_set is not None, "Need a dev set to search hyperparams"	
			best_conf, _ = hypertune(args.lex, args.dev_set, label_map, scorer, 
								hyperparams_grid, args.res_path)
		else:
			best_conf = default_conf
		#run model with the best hyperparams				
		main(args.lex, args.test_set, label_map, args.run_id, best_conf, args.res_path)
		if args.debug is not None:
			print "[DEBUG MODE]"	
			debug(args.lex, args.test_set, label_map, best_conf, args.debug)
	else:
		assert args.cv > 2, "need at leat 2 folds for cross-validation"
		results = []
		cv_results_path = None
		if args.res_path is not None:
			#in CV experiments save the results of each fold in an external file
			fname, _ = os.path.splitext(args.res_path)
			cv_results_path = fname+"_"+os.path.basename(args.test)+"_CV.txt"		
		#loop through cross-validation folds
		for cv_fold in xrange(1, args.cv+1):
			if len(hyperparams_grid) > 0:
				assert args.dev_set is not None, "Need a dev set to search hyperparams"
				best_conf, _ = hypertune(args.lex, args.dev_set, label_map, scorer, 
								hyperparams_grid, args.res_path)
			else:
				best_conf = default_conf
			res = main(args.lex, args.test_set+"_"+str(cv_fold), label_map, args.run_id, best_conf, cv_results_path)
			results.append(res)
		
		accs = [res["acc"] for res in results ]
		f1s = [res["avgF1"] for res in results ]
		
		cv_res = {"acc_mean":round(np.mean(accs),3), \
				"acc_std":round(np.std(accs),3), \
				"avgF1_mean":round(np.mean(f1s),3), \
				"avgF1_std":round(np.std(f1s),3), \
				"model":args.lex, \
				"dataset":os.path.basename(args.test_set), \
				"run_id":args.run_id}
		helpers.print_results(cv_res)
		#save the results of each run 
		if args.res_path is not None:
			cols = ["dataset", "run_id", "model", "acc_mean","acc_std","avgF1_mean","avgF1_std"]
			helpers.save_results(cv_res, args.res_path, columns=cols)
