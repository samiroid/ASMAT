#!/usr/bin/python
'''
Sub-space training 
'''
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle
import argparse
import ast   # to pass literal values
# Local
sys.path.append('code')
import nlse as model   # model used           
# import scipy as sp
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sma_toolkit import kfolds, shuffle_split_idx   
from sma_toolkit.evaluation import FmesSemEval
from sma_toolkit.preprocess import rescale_features
from ipdb import set_trace

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# SUB_SIZES=[3,5,10,15,20]
# LRATES=[0.001,0.01,0.05,0.1,0.5]
# RANDS = [False,True]

SUB_SIZES=[10,15,20,25]
LRATES=[0.01,0.05,0.1,0.5]

# SUB_SIZES=[10]
# LRATES=[0.01]
# RANDS = [False]


def colstr(string, color):
    if color is None:
        cstring = string
    elif color == 'red':
        cstring = "\033[31m" + string  + "\033[0m"
    elif color == 'green':    
        cstring = "\033[32m" + string  + "\033[0m"
    return cstring    

def train_subspace(E, X_train, Y_train, X_dev, Y_dev, lrate, sub_size, n_classes, n_epoch, randomize, output_model, silent=False):
    trainset_size = len(X_train)
    #convert to amenable datatypes
    X_train  = np.array(X_train).astype('int32')
    Y_train  = np.array(Y_train).astype('int32')
    # 
    # load data into gpu
    X = theano.shared(X_train, borrow=True)     
    Y = theano.shared(Y_train, borrow=True)  
    # Create model    
    nn = model.NLSE(None, n_classes, sub_size, E=E)
    # SGD Update rule
    updates = [(pr, pr-lrate*T.grad(nn.F, pr)) for pr in nn.params[1:]]
    # Batch
    i = T.lscalar()            
    # givens = { nn.z0 : x_train[i], nn.y  : Y_train[i] }
    givens = { nn.z0 : X[i], nn.y  : Y[i] }
    # Compile 
    train_batch = theano.function([i], nn.F, updates=updates, givens=givens,allow_input_downcast=True)    
    #dbg = theano.function([i], nn.dbg, givens=givens,allow_input_downcast=True)    
    train_idx = np.arange(trainset_size)
    # TRAIN 
    last_obj = None
    last_f1 = None
    best_f1 = [0, 0]    
    patience = 5
    drops=0
    for e in np.arange(n_epoch):                        
        # Epoch train
        obj = 0 
        n = 0
        if randomize:
            rng.shuffle(train_idx)
        for j in train_idx:                                     
            obj += train_batch(j)                
            # INFO
            if not (n % 10) and not silent:
                print "\rTr %d/%d\t\t" % (n+1, trainset_size),
                sys.stdout.flush()   
            n += 1 
        # Evaluation        
        hat_y = np.zeros(Y_dev.shape, dtype='int32')        
        for j, x, y in zip(np.arange(len(X_dev)), X_dev, Y_dev):
            # Prediction                    
            hat_y[j] = np.argmax(nn.forward(x))        
        # f1 = precision_score(Y_dev,hat_y,average="macro")                         
        f1 = FmesSemEval(hat_y, Y_dev, 1, 2)
        if last_f1:
            if best_f1[0] < f1:
                # Keep best model
                best_f1 = [f1, e+1]
                nn.save(output_model)
                best = '*'
            else:
                best = ''
            delta_f1 = round(f1 - last_f1,3)
            if delta_f1 >= 0:
                delta_str = colstr("+%2.2f" % (delta_f1), 'green')
                drops=0
            elif delta_f1 == 0:
                delta_str = "%2.2f" % (delta_f1)
            else: 
                delta_str = colstr("%2.2f" % (delta_f1), 'red')
                drops+=1
            if obj < last_obj:
                obj_str = colstr("%e" % obj, 'green')
            else:
                obj_str = colstr("%e" % obj, 'red')
            last_obj = obj
        else:
            # First model is best model
            best_f1 = [f1, e+1]
            obj_str = "%e" % obj
            last_obj = obj
            delta_str = "" 
            best = ""
            nn.save(model_path)
        last_f1 = f1
        items = (e+1, n_epoch, obj_str, f1, delta_str, best,drops)
        if not silent: print "Epoch %2d/%2d: %s f1 %.3f %s%s (%d drops)" % items
        if drops > patience:
            print colstr("ran out of patience...", 'red')
    #recover best weights
    nn.load(model_path)        
    
    return nn, best_f1

def get_parser():
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(prog='Trains model')
    parser.add_argument('-train_data', help='train data', type=str, required=True)        
    parser.add_argument('-cv', help='number of crossfolds', type=int)       
    parser.add_argument('-emb_paths', type=str, nargs='+', help='path to embeddings')    
    parser.add_argument('-feat_scale', type=str, choices=['std','unit'], help='feature scaling')       
    # MODEL CONFIG    
    parser.add_argument('-m', help='Path where model is saved', type=str, required=True)    
    parser.add_argument('-res_path', type=str, required=True, help='path to the results folder')    
    parser.add_argument('-s', 
        help='random seed for data shuffling, default is 1234', 
        default=1234, 
        type=int)
    parser.add_argument('-n_epoch', help='number of training epochs', 
        default=20, type=int)
    parser.add_argument('-lrate', help='learning rate', default=0.005, 
        type=float)
    parser.add_argument('-sub_size', help='sub-space size', default=10, 
        type=int)
    parser.add_argument('-randomize', help='randomize each epoch', 
        default=True, type=ast.literal_eval)
    parser.add_argument('-normalize_embeddings', help='Normalize embeddings', 
        default=False, type=ast.literal_eval)
    parser.add_argument('-update_emb', 
        help='Update embeddings', default=False, type=ast.literal_eval)
    parser.add_argument('-update_emb_until_iter', 
        help='Update embeddings until iteration', default=3, type=int)
    
    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()          

    with open(args.train_data) as fid:
        users, labels, usr2idx, lbl2idx = cPickle.load(fid)     
    
    # Model path
    model_name = os.path.splitext(os.path.basename(args.res_path))[0]    
    model_path = args.m+"/"+model_name
    dir_path = os.path.dirname(model_path)        
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)            
    
    vals = (model_name, repr(args.emb_paths), os.path.basename(args.train_data), 
            len(labels), args.res_path)  
    print "[model: %s | embeddings: %s | train @ %s (%d) | results @ %s ]" % vals    

    #compose embeddings
    E = None    
    for f in args.emb_paths:
        feats = np.load(f)          
        if E is None:
            E = feats           
        else:
            E = np.hstack((E,feats))      

    assert E is not None, "No features were extracted"  
    
    
    tr_idx, dev_idx = shuffle_split_idx(labels)     
    U     = np.array([users[idx] for idx in tr_idx])
    U_dev = np.array([users[idx] for idx in dev_idx])    
    Y     = np.array([labels[idx] for idx in tr_idx])
    Y_dev = np.array([labels[idx] for idx in dev_idx])

    # Random seed for epoch shuffle and embedding init
    rng = np.random.RandomState(args.s)                       
    n_classes = len(np.unique(Y))                
    binary_f1s = []
    f1s=[]
    precs  = []   
    for f, (train, test) in enumerate(kfolds(args.cv, len(Y), shuffle=True)):                   
        X_train = U[train]
        Y_train = Y[train]
        X_test  = U[test]
        Y_test  = Y[test]       
        #trainset_size = len(x_train)  
        fold_best_score = -1 
        # list [best_score, configuration, scores]
        fold_best = [-1,[],[]]
        # fold_best_subsize = 10
        # fold_best_lrate = 0.01
        # fold_best_iters=args.n_epoch
        # fold_best_randomize=False
        for sub_size in SUB_SIZES:
            for lrate in LRATES:                  
                #the best model will be saved into "model_path"
                train_subspace(E.T, X_train, Y_train, U_dev, Y_dev, lrate, sub_size, n_classes, args.n_epoch+1, args.randomize, model_path,silent=True)             
                #loading the best model (saved under "model_path")
                test_nn = model.NLSE(None, None, n_classes, model_file=model_path)   
                hat_y = np.zeros(Y_test.shape, dtype='int32')       
                # Prediction    
                for j, x, y in zip(np.arange(len(X_test)), X_test, Y_test):                        
                    hat_y[j]=np.argmax(test_nn.forward(x))                       
                
                f1   = f1_score(Y_test, hat_y, average='macro')         
                prec   = precision_score(Y_test, hat_y, average='macro')                
                binary_f1   = FmesSemEval(hat_y, Y_test, 1, 2)            
                
                if binary_f1 > fold_best[0]:                        
                    fold_best[0]=binary_f1
                    fold_best[1]=[sub_size,lrate]                    
                    fold_best[2]=[f1,binary_f1,prec]
                    sys.stdout.write("\r Best Conf >> sub_size: %d | lrate: %.5f | binary f1: %.3f" % (sub_size,lrate,binary_f1))                                        
                    sys.stdout.flush()                            
        f1, binary_f1, prec = fold_best[2]
        fold_best_subsize, fold_best_lrate = fold_best[1]        
        precs += [prec]                 
        f1s += [f1]                     
        binary_f1s += [binary_f1]                   
        vals = (model_name, f, args.cv, fold_best_subsize, fold_best_lrate, binary_f1, f1,  prec)            
        print "\nNLSE (%s) [%d\%d] ** best conf: {s: %d; lr: %.5f} | binary f1: %.3f | f1: %.3f | prec: %.3f" % vals
    
    print "*"*90
    vals = (model_name, np.mean(binary_f1s), np.std(binary_f1s), 
                        np.mean(f1s), np.std(f1s), 
                        np.mean(precs), np.std(precs))
    print "* NLSE (%s) > avg binary f1: %.3f (%.3f) | avg f1: %.3f (%.3f) | avg prec: %.3f (%.3f) |  " % vals
    print "*"*90
    
    dirname = os.path.dirname(args.res_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)            
    
    with open(args.res_path,"w") as fod:
        columns = "size,model,binary_f1,std_binary_f1,f1,std_f1,prec,std_prec\n"
        fod.write(columns)      
        vals = [len(Y)] + list(vals)        
        fod.write("%d, %s, %.3f, %.3f, %.3f, %.3f,%.3f, %.3f" % tuple(vals))    


