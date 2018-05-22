import argparse
import csv
import sys
import os 
csv.field_size_limit(sys.maxsize)

import sklearn 
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd 
import numpy as np 

import gensim 
from gensim.models import Word2Vec
from ipdb import set_trace
import CNN_text

from ASMAT.lib import embeddings, vectorizer, data, helpers


def main(train, test, dev, embs_path, total_epochs=10, weights_file=None, results_path=None):
    print "[reading data]"
    train_data = data.read_dataset(train)
    train_docs = [x[1] for x in train_data]
    train_Y = [x[0] for x in train_data]

    test_data = data.read_dataset(test)
    test_docs = [x[1] for x in test_data]
    test_Y = [x[0] for x in test_data]
    
    dev_data = data.read_dataset(dev)
    dev_docs = [x[1] for x in dev_data]
    dev_Y = [x[0] for x in dev_data]
    
    #convert labels to one-hot
    label_map = vectorizer.build_vocabulary(test_Y+train_Y+dev_Y)
    train_Y = vectorizer.one_hot(label_map, train_Y)
    dev_Y = vectorizer.one_hot(label_map, dev_Y)
    test_Y = vectorizer.one_hot(label_map, test_Y)
    #convert to argmax
    test_Y = np.argmax(test_Y, axis=1) 
    n_labels = len(train_Y[0])   
    print "[loading embeddings]"
    wvs = embeddings.embeddings_to_dict(embs_path)    
    # preprocessor for texts    
    print "[preprocessing...]"
    all_docs = train_docs + test_docs + dev_docs
    max_len = max([len(x.split()) for x in all_docs])
    print "[max len: {}]".format(max_len)
    p = CNN_text.Preprocessor(max_features=len(wvs), maxlen=max_len, wvs=wvs)        
    p.preprocess(all_docs)
    train_X = p.build_sequences(train_docs)
    test_X = p.build_sequences(test_docs)
    dev_X = p.build_sequences(dev_docs)
    # then the CNN
    cnn = CNN_text.TextCNN(p, n_labels=n_labels, filters=[2,3], n_filters=50, dropout=0.0)     

    if weights_file:
        cnn.model.load_weights('weights.hdf5')

    epochs_per_iter = 1
    epochs_so_far = 0
    print "training"
    while epochs_so_far < total_epochs:
        cnn.train(train_X, train_Y, nb_epoch=epochs_per_iter, X_val=dev_X, y_val=dev_Y)
        epochs_so_far += epochs_per_iter        
        Y_hat = cnn.predict(dev_X)         
        acc = accuracy_score(np.argmax(dev_Y, axis=1), Y_hat) 
        avgF1 = f1_score(np.argmax(dev_Y, axis=1), Y_hat, average="macro") 
        res={"acc":round(acc,3), \
            "avgF1":round(avgF1,3)}
        helpers.print_results(res)
        #print("acc @ epoch %s: %s" % (epochs_so_far, acc))    
    
    Y_hat = cnn.predict(test_X)            
    acc = accuracy_score(test_Y, Y_hat) 
    avgF1 = f1_score(test_Y, Y_hat,average="macro") 
            
    results = {"acc":round(acc,3), \
            "avgF1":round(avgF1,3), \
            "model":"CNN", \
            "dataset":os.path.basename(test), \
            "run_id":"NEURAL"}    
    helpers.print_results(results)        
    if results_path is not None:
        cols = ["dataset", "model", "run_id", "acc", "avgF1"]
        helpers.save_results(results,results_path, cols, sep="\t")

def get_argparser():
    parser = argparse.ArgumentParser(prog='CNN model trainer')
    parser.add_argument('-m', help='Path where model is saved', type=str)
    parser.add_argument('-res_path', required=True,type=str, help='training data')
    parser.add_argument('-train', required=True,type=str, help='training data')
    parser.add_argument('-dev', required=True,type=str, help='dev data')
    parser.add_argument('-test', required=True,type=str, help='dev data')    
    parser.add_argument('-emb', type=str, help='embeddings')    
    parser.add_argument('-epochs', type=int, default=5, help='number of epochs')    
    args = parser.parse_args(sys.argv[1:])
    return args

if __name__ == '__main__':
    args = get_argparser()
    main(train=args.train, test=args.test, dev=args.dev, total_epochs=args.epochs,
         embs_path=args.emb, results_path=args.res_path)

