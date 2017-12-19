import argparse
import cPickle
from collections import defaultdict
from ipdb import set_trace
import math
import numpy as np
import os
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sys

sys.path.append("../ASMAT")
from ASMAT.models import Lexicon_SMA
from ASMAT.lib.data import read_dataset
from ASMAT.lib import helpers

def get_parser():
    parser = argparse.ArgumentParser(description="Lexicon Classifier")    
    #Basic Input
    parser.add_argument('-lex', type=str, required=True, help='lexicon file')
    parser.add_argument('-decision_threshold', type=float, default=0.55, \
						help='scores above are considered positive, else negative')
    parser.add_argument('-neutral_threshold', type=float, help='threshold for neutral class')    
    parser.add_argument('-pos_label', type=str, default="positive", \
						help='label for the positive class')
    parser.add_argument('-neg_label', type=str, default="negative",
						help='label for the negative class')
    parser.add_argument('-ts', type=str, required=True, \
						help='test file(s)')
    parser.add_argument('-res', type=str, required=True, \
						help='results file')
    parser.add_argument('-run_id', type=str, help='run id')

    return parser

if __name__=="__main__":	
	parser = get_parser()
	args = parser.parse_args()  	

	label_map = {args.pos_label: 1, args.neg_label: 0}
	model = Lexicon_SMA(path=args.lex, score_mean=True)	
	dt = read_dataset(args.ts, labels=label_map.keys())
	X = [x[1] for x in dt]
	Y = [label_map[x[0]] for x in dt]
	scores = model.predict_text_many(X)		
	Y_hat = map(lambda x:1 if x>=args.decision_threshold else 0, scores)		
	avgF1 = f1_score(Y, Y_hat,average="macro") 		
	acc = accuracy_score(Y, Y_hat)				
	# print "%s ** acc: %.3f | F1: %.3f " % (args.ts, acc, avgF1)

	run_id = args.run_id
	if run_id is None:
	    run_id = os.path.splitext(os.path.basename(args.lex))[0]
	fname = os.path.basename(args.ts)
	results = {"acc": round(acc, 3),
               "avgF1": round(avgF1, 3),
               "data": repr(args.ts),
            	"lex": args.lex,
            	">": fname,
            	"run_id": run_id}
	
	helpers.print_results(results, columns=[">", "run_id", "acc", "avgF1"])
	if args.res is not None:
		helpers.save_results(results, args.res, columns=["run_id", "acc", "avgF1"])



# with open(test_file) as fid:
		# 	for l in fid:
		# 		splt = l.replace("\n","").split("\t")
		# 		# set_trace()
		# 		try:
		# 			y = label_map[splt[0]]
		# 		except KeyError:
		# 			pass
		# 		Y_test.append(y)
		# 		txt = splt[1]
		# 		score = model.predict_text_one(txt)[0]
		# 		if math.isnan(score): score = 0
		# 		scores.append(score)
		# 		# print txt, " | ", score
