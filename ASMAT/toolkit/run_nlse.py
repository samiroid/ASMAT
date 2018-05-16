'''
Sub-space training 
'''

import ast   # to pass literal values
import argparse
from collections import defaultdict
import cPickle
from ipdb import set_trace
import numpy as np
import os
from sklearn.metrics import f1_score, accuracy_score
import sys

#local 
sys.path.append("..")
from ASMAT.lib import helpers, embeddings, vectorizer, data
from ASMAT.models.nlse import nlse

def get_argparser():
    parser = argparse.ArgumentParser(prog='NLSE model runner')
    parser.add_argument('-model_path', help='Path where model is saved', type=str, required=True)
    parser.add_argument('-data_path', required=True,type=str, help='training data')    
    parser.add_argument('-res_path', type=str, help='results file')    
    args = parser.parse_args(sys.argv[1:])
    return args

if __name__ == '__main__':
    # ARGUMENT HANDLING    
    args = get_argparser()       
    clf = nlse.load_model(args.model_path)
    dataset = data.read_dataset(args.data_path)    
    docs = [x[1] for x in dataset]
    labels = [x[0] for x in dataset]    
    X, _ = vectorizer.docs2idx(docs, clf.vocab)    
    #map numeric labels to text
    inv_label_map = {ix:label for label,ix in clf.label_map.items()}    
    y_hat = [inv_label_map[y] for y in clf.predict(X)]    
    with open(args.res_path,"w") as fod:
        for y, x in zip(labels, y_hat):
            fod.write("{}\t{}\n".format(x,y))
