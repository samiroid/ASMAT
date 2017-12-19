#!/usr/bin/python
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
import theano
import theano.tensor as T

#local 
sys.path.append("..")
import nlse 
from ASMAT.lib import helpers

def colstr(string, color):
    if color is None:
        cstring = string
    elif color == 'red':
        cstring = "\033[31m" + string  + "\033[0m"
    elif color == 'green':    
        cstring = "\033[32m" + string  + "\033[0m"
    return cstring    

def evaluate(model, X, Y):
    # Evaluation
    cr = 0.
    mapp = np.array([1, 2, 0])
    ConfMat = np.zeros((3, 3))
    Y_hat = np.zeros(len(Y), dtype='int32')
    # dev_p_y = np.zeros((3, dev_y.shape[0]))
    for j, x, y in zip(np.arange(len(X)), X, Y):
        # Prediction
        x = np.array(x)
        p_y = model.forward(x)
        hat_y = np.argmax(p_y)
        Y_hat[j]=hat_y
        # Confusion matrix
        ConfMat[mapp[y], mapp[hat_y]] += 1
        # Accuracy
        cr = (cr*j + (hat_y == y).astype(float))/(j+1)

    avgF1 = f1_score(Y, Y_hat,average="macro")        
    acc = accuracy_score(Y, Y_hat)            
    # binary_f1 = helpers.FmesSemEval(Y_hat, Y)        

    return avgF1, acc

def train_nlse(train_x, train_y, vocabulary, E, st, ed, args):
    # Random seed for epoch shuffle and embedding init
    rng = np.random.RandomState(args.rand_seed)
    # Weighted confusion matrix cost
    if args.neutral_penalty != 1:
        weigthed_CM = weighted_confusion_matrix(
            neut_penalty=args.neutral_penalty)
    else:
        weigthed_CM = None

    n_sent_train = len(st)
    # Create model
    nn = nlse.NN(E, args.sub_size, weight_CM=weigthed_CM,
                 init=args.rand_seed)
    # Ensure types compatible with GPU
    train_x = train_x.astype('int32')
    train_y = train_y.astype('int32')
    st = st.astype('int32')
    ed = ed.astype('int32')
    # Store as shared variables (push into the GPU)
    train_x = theano.shared(train_x, borrow=True)
    train_y = theano.shared(train_y, borrow=True)
    st = theano.shared(st, borrow=True)
    ed = theano.shared(ed, borrow=True)
    # SGD Update rule
    E = nn.params[0]
    # Sub-space: Do not update E
    updates = [(pr, pr - args.lrate * T.grad(nn.F, pr))
               for pr in nn.params[1:]]
    # Batch
    i = T.lscalar()
    givens = {nn.z0: train_x[st[i]:ed[i], 0], nn.y: train_y[i]}
    # Compile
    train_batch = theano.function([i], nn.F, updates=updates, givens=givens)
    train_idx = np.arange(n_sent_train).astype('int32')
    # TRAIN
    last_obj = None
    last_Fm = None
    best_Fm = [0, 0]
    last_Acc = None
    stop = False
    for i in np.arange(args.n_epoch):
        # Epoch train
        obj = 0
        n = 0
        if args.randomize:
            rng.shuffle(train_idx)
        for j in train_idx:
            obj += train_batch(j)
            # INFO
            if not n % 10:
                print "\rEpoch: %d | Training %d/%d %s" % (i + 1, n + 1, n_sent_train, " " * 20),
                sys.stdout.flush()
            n += 1
        Fm, cr = evaluate(nn, dev_x, dev_y)

        # INFO
        if last_Fm:
            if best_Fm[0] < Fm:
                # Keep best model
                best_Fm = [Fm, i + 1]
                nn.save(args.m)
                best = '*'
            else:
                best = ''
            delta_Fm = Fm - last_Fm
            if delta_Fm >= 0:
                delta_str = colstr("+%2.2f" % (delta_Fm * 100), 'green')
            else:
                delta_str = colstr("%2.2f" % (delta_Fm * 100), 'red')
            if obj < last_obj:
                obj_str = colstr("%e" % obj, 'green')
            else:
                obj_str = colstr("%e" % obj, 'red')
            last_obj = obj
        else:
            # First model is best model
            best_Fm = [Fm, i + 1]
            obj_str = "%e" % obj
            last_obj = obj
            delta_str = ""
            best = ""
            nn.save(args.m)
        if last_Acc:
            if last_Acc > cr:
                acc_str = "Acc " + colstr("%2.2f%%" % (cr * 100), 'red')
            else:
                acc_str = "Acc " + colstr("%2.2f%%" % (cr * 100), 'green')
        else:
            acc_str = "Acc %2.2f%%" % (cr * 100)
        last_Acc = cr
        last_Fm = Fm
        items = (i + 1, args.n_epoch, obj_str,
                 acc_str, Fm * 100, delta_str, best)
        # logging.info("Epoch %2d/%2d: %s %s Fm %2.2f%% %s%s" % items)
        if args.log is not None:
            logging.info("%s,%.3f,%.3f" % (i + 1, cr, Fm))
        else:
            print "Epoch %2d/%2d: %s %s Fm %2.2f%% %s%s" % items

    #load best model
    nn.load(args.m)
    return nn

def weighted_confusion_matrix(pos_penalty=0, neg_penalty=0, neut_penalty=0):
    """
    """
    weigthed_CM = np.zeros((3, 3))
    weigthed_CM[0, :] = np.array([1, 0, pos_penalty ])  # positive
    weigthed_CM[1, :] = np.array([0, 1, neg_penalty ])  # negative
    weigthed_CM[2, :] = np.array([0, 0, neut_penalty])  # neutral
    # Normalize
    weigthed_CM = weigthed_CM * 3. / weigthed_CM.sum()
    weigthed_CM = weigthed_CM.astype(theano.config.floatX)
    weigthed_CM = theano.shared(weigthed_CM, borrow=True)
    
    return weigthed_CM

if __name__ == '__main__':
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(prog='Trains model')
    # parser.add_argument('-o', help='Folder where the train data embeddings are', 
    #     type=str, required=True)
    # parser.add_argument('-e', help='Original embeddings file', type=str, 
    #     required=True)
    parser.add_argument('-m', help='Path where model is saved', type=str, required=True)
    parser.add_argument('-tr', required=True,type=str, help='training data')
    parser.add_argument('-dev', required=True,type=str, help='dev data')
    parser.add_argument('-ts', required=True,type=str, help='dev data')
    parser.add_argument('-emb', required=True,type=str, help='embedding path')
    parser.add_argument('-log', type=str, help='logger path')    
    parser.add_argument('-run_id', type=str, help='run id')
    parser.add_argument('-res_path', type=str, help='results file')
    # CONFIG    
    parser.add_argument('-rand_seed', default=1234, type=int, help='random seed for data shuffling, default is 1234')
    parser.add_argument('-n_epoch', help='number of training epochs', default=12, type=int)
    parser.add_argument('-lrate', help='learning rate', default=0.01, type=float)
    parser.add_argument('-sub_size', help='sub-space size', default=10, type=int)
    parser.add_argument('-randomize', help='randomize each epoch', default=True, type=ast.literal_eval)
    parser.add_argument('-neutral_penalty', help='Penalty for neutral cost', default=0.25, type=float)
    args = parser.parse_args(sys.argv[1:])
    
    
    # logging.info('Initialized model from embeddings %s' % args.emb)        
    # logging.info("Training data: %s" % args.tr)
    # logging.info("Dev data: %s" % args.dev)
    with open(args.tr, 'rb') as fid:     
        train_x, train_y, vocabulary, label_map, E, st, ed = cPickle.load(fid) 
        E = E.astype(theano.config.floatX)    
    with open(args.dev, 'rb') as fid:
        dev_x, dev_y,_,_ = cPickle.load(fid) 
    with open(args.ts, 'rb') as fid:
        ts_x, ts_y,_,_ = cPickle.load(fid) 
    
    nn = train_nlse(train_x, train_y, vocabulary, E, st, ed, args)    
    avgF1, acc = evaluate(nn, ts_x, ts_y)
    fname = os.path.basename(args.ts) 
    run_id = args.run_id
    if run_id is None: run_id = "NLSE"
    results = {"acc":round(acc,3),
               "avgF1":round(avgF1,3),
               # "binary_f1":round(binary_f1,3),
               "subsize":args.sub_size,
               "lrate":args.lrate,
               "dataset":fname,
               "run_id":run_id}

    helpers.print_results(results,columns=["dataset","run_id","lrate","subsize","acc","avgF1"])
    if args.res_path is not None:
        helpers.save_results(results, args.res_path, columns=["dataset","run_id","lrate","subsize","acc","avgF1"])
