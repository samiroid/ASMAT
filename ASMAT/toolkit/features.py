import os
import codecs
import cPickle
import argparse

from ipdb import set_trace
import numpy as np

import sys
sys.path.append("..")

from ASMAT.lib.extract import docs2idx, build_vocabulary
from ASMAT.lib import embeddings, features
from ASMAT.lib.data import read_dataset, flatten_list

# from ASMAT.toolkit.extract import docs2idx, word2idx
# from ASMAT.toolkit import features
# from ASMAT.toolkit import embeddings 


def get_parser():
	par = argparse.ArgumentParser(description="Extract Features")
	par.add_argument('-input', type=str, required=True, nargs='+', help='train data')  
	par.add_argument('-out_folder', type=str, required=True, help='output folder')
	par.add_argument('-vectors', type=str, help='path to embeddings')          
	par.add_argument('-bow', type=str, choices=['bin', 'freq'], nargs='+', help='bow features')
	par.add_argument('-boe', type=str, choices=['bin', 'sum'], nargs='+', help='boe features')             
	par.add_argument('-nlse', action="store_true")
	return par

if __name__ == "__main__":	
	parser = get_parser()
	args = parser.parse_args()	
	assert args.bow is not None or args.boe is not None or args.nlse, "please, specify some features"
	if args.boe is not None or args.nlse:
		assert args.vectors is not None, "missing vectors"

	#create output folder if needed
	if not os.path.exists(os.path.dirname(args.out_folder)):
	    os.makedirs(os.path.dirname(args.out_folder))   	
	
	for dataset in args.input:
		print "[extracting features @ {}]".format(repr(dataset))
		E = None
		with open(dataset,"rb") as fid:
			X, Y, vocabulary, label_map = cPickle.load(fid)
			basename = os.path.splitext(os.path.basename(dataset))[0]
			path = args.out_folder
			if args.bow is not None:
				for agg in args.bow:
					fname = basename+"_BOW_"+agg							
					print "\t > BOW ({})".format(fname)
					bow = features.BOW(X, vocabulary, agg=agg)
					np.save(args.out_folder+fname, bow)						
					# print "\t > {}".format(repr(bow))
					# print len(X), bow.shape
			if args.boe is not None:
				for agg in args.boe:
					fname = basename+"_BOE_"+agg							
					print "\t > BOE ({})".format(fname)
					E, _ = embeddings.read_embeddings(args.vectors,wrd2idx=vocabulary)
					boe = features.BOE(X, E, agg=agg)
					# print boe.shape
					np.save(args.out_folder+fname, boe)						
			if args.nlse:
				fname = basename+"_NLSE.pkl"
				X, Y, st, ed = features.NLSE(X, Y)				
				E, _ = embeddings.read_embeddings(args.vectors,wrd2idx=vocabulary)
				with open(args.out_folder+fname,"w") as fod: 
					cPickle.dump([X, Y, vocabulary, label_map, E, st, ed], fod, cPickle.HIGHEST_PROTOCOL)	
