#!/usr/bin/env python

"""
Training a convolutional network for sentence classification,
as described in paper:
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf
"""
import os
import cPickle as pickle
import numpy as np
import theano
import sys
import argparse
import warnings
from ipdb import set_trace
import itertools
warnings.filterwarnings("ignore")   
import pprint
sys.path.append(".")
from conv_net_classes import *
from process_data import process_data
#from evaluation import FmesSemEval, accuracy
from sklearn.metrics import f1_score, accuracy_score
sys.path.append("..")
from ASMAT.lib import helpers

def get_idx_from_sent(sent, word_index, max_l, pad):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    Drop words non in word_index. Attardi.
    :param max_l: max sentence length
    :param pad: pad length
    """
    x = [0] * pad                # left padding
    words = sent.split()[:max_l] # truncate words from test set
    for word in words:
        if word in word_index: # FIXME: skips unknown words
            x.append(word_index[word])
    while len(x) < max_l + 2 * pad: # right padding
        x.append(0)
    return x


def make_idx_data_cv(revs, word_index, cv, max_l, pad):
    """
    Transforms sentences into a 2-d matrix and splits them into
    train and test according to cv.
    :param cv: cross-validation step
    :param max_l: max sentence length
    :param pad: pad length
    """
    
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_index, max_l, pad)
        sent.append(rev["y"])
        if rev["split"] == cv:
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train, dtype="int32")
    test = np.array(test, dtype="int32")
    return train, test
  
def read_corpus(filename, word_index, max_l, label_dict, pad=2, textField=3,tagField=0):
    test, y = [], []
    with open(filename) as f:
        for line in f:
            fields = line.strip().split("\t")
            text = fields[textField]            
            text_clean = text.lower()
            label = label_dict.index(fields[tagField]) 
            sent = get_idx_from_sent(text_clean, word_index, max_l, pad)
            #sent.append(0)      # unknown y
            test.append(sent)
            y.append(label)            
    return np.array(test, dtype="int32"), y
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="CNN sentence classifier.")
    
    parser.add_argument('model', type=str, default='mr',
                        help='model file (default %(default)s)')
    parser.add_argument('input', nargs='+', type=str,
                        help='train/test file in SemEval twitter format')
    parser.add_argument('-tagField', type=int, default=1,
                        help='label field in files (default %(default)s)')
    parser.add_argument('-textField', type=int, default=2,
                        help='text field in files (default %(default)s)')
    parser.add_argument('-res_path', type=str, 
                        help='path to results file')

    args = parser.parse_args()
    # test
    with open(args.model) as mfile:
        cnn = ConvNet.load(mfile)
    
    with open(args.model+"_aux") as mfile:
        word_index, labels, max_l, pad = pickle.load(mfile)

    cnn.activations = ["relu"] #TODO: save it in the model
    # pos_idx = labels.index("1")
    # neg_idx = labels.index("-1")
    tagField = args.tagField
    textField = args.textField

    for dataset in args.input:
        test_set_x, test_set_y = read_corpus(dataset, word_index, 
                                             max_l, labels, 
                                             pad, textField=textField,
                                             tagField=tagField)
        test_set_y_pred = cnn.predict(test_set_x)
        test_model = theano.function([cnn.x], test_set_y_pred, allow_input_downcast=True)
        results = test_model(test_set_x)
        acc = accuracy_score(np.array(test_set_y), results)
        fmes = f1_score(test_set_y, results,average="macro")
        print "Evaluation %s > acc: %.3f | fmes: %.3f" % (dataset, acc,fmes)
        results = {"acc":round(acc,3), \
			"avgF1":round(fmes,3),	\
			"model":"CNN", \
			"dataset":os.path.basename(dataset), \
			"run_id":"NEURAL", \
			}
	cols = ["dataset", "run_id", "acc", "avgF1","hyper"]
	helpers.print_results(results, columns=cols)
	if args.res_path is not None:
		cols = ["dataset", "model", "run_id", "acc", "avgF1"]
		helpers.save_results(results, args.res_path, sep="\t", columns=cols)
    
    # invert indices (from process_data.py)
    # for line, y in zip(open(args.input), results):
    #     tokens = line.split("\t")
    #     tokens[tagField] = labels[y]
    #     print "\t".join(tokens),
    
