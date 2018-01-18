import itertools
import json
import numpy as np
import os
import pandas as pd
from pdb import set_trace
import pprint
import string

def str2seed(s):
	"""
		simple heuristic to convert strings into a digit
		map each character to its index in the alphabet (e.g. 'a'->1, 'b'->2)
	"""
	assert type(s) == str, "input is not a string"
	seed = ""
	for c in s.lower():		
		try:
			z = string.ascii_letters.index(c)+1	
		except ValueError:
			z=0
		seed += str(z)
	#limit the seed to 9 digits
	return int(seed[:9])

def save_results(results, path, columns=None):

	if columns is not None:
		row = [str(results[c]) for c in columns if c in results]
	else:
		columns = results.keys()
		row = [str(v) for v in results.values()]

	dirname = os.path.dirname(path)
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	if not os.path.exists(path):
		with open(path, "w") as fod:
			fod.write(",".join(columns) + "\n")
			fod.write(",".join(row) + "\n")
	else:
		with open(path, "a") as fod:
			fod.write(",".join(row) + "\n")


def print_results(results, columns=None):
	if columns is not None:
		row = [results[c] for c in columns if c in results]
	else:
		columns = results.keys()
		row = results.values()
	s = ["{}: {}| ".format(c, r) for c, r in zip(columns, row)]
	print "** " + "".join(s)

def read_results(path, metric="avgF1", models=None, datasets=None):
	"""
		read results:
		assumes the following columns: dataset, model, [metric]
	"""
	c = pd.read_csv(path)		
	#get datasets 
	if datasets is not None:
		c = c[c["dataset"].isin(datasets)]		
	if models is not None:
		c = c[c["model"].isin(models)]		
	uniq_models = c["model"].unique().tolist()
	uniq_datasets = c["dataset"].unique().tolist()	
	dt = [[d] + c[c["dataset"]==d][metric].values.tolist() for d in uniq_datasets]
	dt = dt
	columns = ["dataset"] + uniq_models	
	df = pd.DataFrame(dt,columns=columns)	
	if models is not None:
		#re-order model columns
		df = df[["dataset"]+models]
	return df

def get_hyperparams(path, default_conf):
	confs = json.load(open(path, 'r'))
	#add configurations that are not specified with the default values
	confs = dict(default_conf.items() + confs.items())
	params = confs.keys()
	choices = [x if isinstance(x,list) else [x] for x in confs.values()]
	combs = list(itertools.product(*choices))	
	hyperparams = [{k:v for k,v in zip(c,x)} for x, c in zip(combs, [params]*len(combs))]
	return hyperparams
