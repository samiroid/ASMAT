import os
import codecs
import cPickle
import argparse

from ipdb import set_trace
import numpy as np

import sys
sys.path.append("..")

from ASMAT.lib.vectorizer import docs2idx, build_vocabulary
from ASMAT.lib import embeddings, features
from ASMAT.lib.data import read_dataset, flatten_list

def run(inputs, opts):
	for dataset in inputs:
		print "[extracting features @ {}]".format(repr(dataset))
		E = None
		with open(dataset, "rb") as fid:
			X, Y, vocabulary, label_map = cPickle.load(fid)
			basename = os.path.splitext(os.path.basename(dataset))[0]			
			if opts.bow is not None:
				for agg in opts.bow:
					fname = basename + "-BOW-" + agg.upper()
					print "\t > BOW ({})".format(fname)
					if agg == "bin":
						bow = features.BOW(X, vocabulary)
					elif agg == "freq":
						bow = features.BOW_freq(X, vocabulary)
					np.save(opts.out_folder + fname, bow)
					# print "\t > {}".format(repr(bow))
					# print len(X), bow.shape
			if opts.boe is not None:
				for agg in opts.boe:
					fname = basename + "-BOE-" + agg.upper()
					print "\t > BOE ({})".format(fname)
					E, _ = embeddings.read_embeddings(opts.embeddings, wrd2idx=vocabulary)
					boe = features.BOE(X, E, agg=agg)
					# print boe.shape
					np.save(opts.out_folder + fname, boe)
			if opts.nlse:
				fname = basename + "_NLSE.pkl"
				# X, Y, st, ed = features.NLSE(X, Y)
				E, _ = embeddings.read_embeddings(opts.embeddings, wrd2idx=vocabulary)
				np.save(fname, E)
				# with open(opts.out_folder + fname, "w") as fod:
				# 	cPickle.dump([X, Y, vocabulary, label_map, E, st, ed],
				# 	             fod, cPickle.HIGHEST_PROTOCOL)


def get_parser():
	par = argparse.ArgumentParser(description="Extract Features")
	par.add_argument('-input', type=str, required=True, nargs='+', help='train data')
	par.add_argument('-out_folder', type=str, required=True, help='output folder')
	par.add_argument('-bow', type=str, choices=['bin', 'freq'], nargs='+', help='bow features')
	par.add_argument('-boe', type=str, choices=['bin', 'sum'], nargs='+', help='boe features')
	par.add_argument('-nlse', action="store_true")
	par.add_argument('-cv', type=int, help='crossfold')
	par.add_argument('-cv_from', type=str, nargs='*', \
					help="files for crossvalidation")
	par.add_argument('-embeddings', type=str, help='path to embeddings')

	return par

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()
	assert args.bow is not None or args.boe is not None or args.nlse, "please, specify some features"
	if args.boe is not None or args.nlse:
		assert args.embeddings is not None, "missing embeddings"

	#create output folder if needed
	args.out_folder = args.out_folder.rstrip("/") + "/"
	if not os.path.exists(os.path.dirname(args.out_folder)):
	    os.makedirs(os.path.dirname(args.out_folder))

	#loop through cross-validation folds (if any)
	if args.cv is None:
		fnames = args.input
		run(fnames, args)
	else:
		assert args.cv > 2, "need at leat 2 folds for cross-validation"
		for cv_fold in xrange(1, args.cv+1):
			if args.cv_from is None:
				cv_fnames = [f+"_"+str(cv_fold) for f in args.input]
			else:
				cv_fnames = [f + "_" + str(cv_fold) for f in args.cv_from]
			run(cv_fnames, args)			

	
