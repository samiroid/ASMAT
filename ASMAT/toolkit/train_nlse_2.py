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
from ASMAT.lib import helpers, embeddings, vectorizer
from ASMAT.models.nlse import nlse

def hypertune(train, dev, emb_path, obj, hyperparams, res_path=None):    		

    with open(train, 'rb') as fid:     
        X_train, Y_train, vocabulary, _ = cPickle.load(fid)         
    with open(dev, 'rb') as fid:
        X_dev, Y_dev,_,_ = cPickle.load(fid)     
    E, _ = embeddings.read_embeddings(emb_path, wrd2idx=vocabulary)
    
    best_hp = None
    best_score = 0
    for hp in hyperparams:
        #initialize model with the hyperparameters	
        nn = nlse.NLSE(E, **hp)
        nn.fit(X_train, Y_train, X_dev, Y_dev, silent=False)
        Y_hat = nn.predict(X_dev)        
        score = obj(Y_dev, Y_hat)
        print "[score: {} | hyperparameters: {}]".format(score, repr(hp))
        if score > best_score:
            best_score = score
            best_hp = hp
        results = {"score":round(score,3), "hyper":repr(hp)}
        if res_path is not None:
            helpers.save_results(results,res_path, sep="\t")
        helpers.print_results(results)
    print ""
    print "[best conf: {} | score: {}]".format(repr(best_hp),best_score)
    return best_hp, best_score

def hypertuner(train, dev, test, emb_path, obj, hyperparams, run_id, 
             res_path=None, model_path=None):      
    with open(train, 'rb') as fid:     
        X_train, Y_train, vocabulary = cPickle.load(fid)         
    with open(dev, 'rb') as fid:
        X_dev, Y_dev, _ = cPickle.load(fid) 
    with open(test, 'rb') as fid:
        X_test, Y_test, _ = cPickle.load(fid) 
    E, _ = embeddings.read_embeddings(emb_path, wrd2idx=vocabulary)
    Ys = Y_train + Y_dev + Y_test
    label_map = vectorizer.build_vocabulary(Ys)
    Y_train = [label_map[y] for y in Y_train]
    Y_test = [label_map[y] for y in Y_test]
    Y_dev = [label_map[y] for y in Y_dev]
    
    dataset = os.path.basename(test)     
    best_hp = None
    best_score = 0
    best_results = None

    for hp in hyperparams:
        #initialize model with the hyperparameters	
        nn = nlse.NLSE(E, label_map=label_map, vocab=vocabulary, **hp)
        # nn = nlse.NLSE(E, **hp)
        nn.fit(X_train, Y_train, X_dev, Y_dev, silent=False)
        Y_hat = nn.predict(X_test)        
        score = obj(Y_test, Y_hat)
        print "[score: {} | hyperparameters: {}]".format(score, repr(hp))
        if score > best_score:
            if model_path is not None:
                nn.save(model_path)
            best_score = score
            best_hp = hp
            acc = accuracy_score(Y_test, Y_hat)
            avgF1 = f1_score(Y_test, Y_hat,average="macro") 
            rep_hp = {p:hp[p] for p in ["sub_size","lrate"]}            
            best_results = {"acc":round(acc,3), \
                    "avgF1":round(avgF1,3), \
                    "model":"NLSE", \
                    "dataset":dataset, \
                    "run_id":run_id,
                    "hyper":repr(rep_hp)}
        res = {"score":round(score,3), "hyper":repr(hp)}        
        helpers.print_results(res)
        
    if res_path is not None:
        cols = ["dataset", "model", "run_id", "acc", "avgF1"]
        helpers.save_results(best_results,res_path, cols, sep="\t")
    print ""
    print "[best conf: {} | score: {}]".format(repr(best_hp),best_score)
    return best_hp, best_score

def main(train, dev, test, emb_path, hyperparams, run_id=None, res_path=None):
    with open(train, 'rb') as fid:     
        X_train, Y_train, vocabulary, _ = cPickle.load(fid)         
    with open(dev, 'rb') as fid:
        X_dev, Y_dev,_,_ = cPickle.load(fid) 
    with open(test, 'rb') as fid:
        test_x, test_y,_,_ = cPickle.load(fid) 
    E, _ = embeddings.read_embeddings(emb_path, wrd2idx=vocabulary)
    nn = nlse.NLSE(E, **hyperparams)
    nn.fit(X_train, Y_train, X_dev, Y_dev)
    y_hat = nn.predict(test_x)
    avgF1 = f1_score(test_y, y_hat,average="macro") 		
    acc = accuracy_score(test_y, y_hat)					        
    run_id = run_id
    dataset = os.path.basename(test)     
    hp = {p:hyperparams[p] for p in ["sub_size","lrate"]}    
    if run_id is None: run_id = "NLSE"
    results = {"acc":round(acc,3), \
                "avgF1":round(avgF1,3), \
                "model":"NLSE", \
                "dataset":dataset, \
                "run_id":run_id,
                "sub_size":hyperparams["sub_size"],
                "lrate":hyperparams["lrate"]}
    cols = ["dataset", "model", "run_id", "acc", "avgF1","sub_size"]
    helpers.print_results(results,columns=["dataset","run_id","lrate","subsize","acc","avgF1"])
    if res_path is not None:
        helpers.save_results(results, res_path, sep="\t", columns=cols)
    return results

def get_argparser():
    parser = argparse.ArgumentParser(prog='NLSE model trainer')
    parser.add_argument('-m', help='Path where model is saved', type=str, required=True)
    parser.add_argument('-train', required=True,type=str, help='training data')
    parser.add_argument('-dev', required=True,type=str, help='dev data')
    parser.add_argument('-test', required=True,type=str, help='dev data')
    parser.add_argument('-emb_path', required=True,type=str, help='embedding path')
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
    parser.add_argument('-hyperparams_path', type=str, default="", help='path to a dictionary of hyperparameters')
    parser.add_argument('-cv', type=int, help='crossfold')
    parser.add_argument('-patience', type=int, default=10, help='patience')    
    args = parser.parse_args(sys.argv[1:])
    return args

if __name__ == '__main__':
    # ARGUMENT HANDLING    
    args = get_argparser()       

    conf = {"sub_size":args.sub_size, \
            "lrate": args.lrate, \
            "rand_seed": args.rand_seed, 
            "n_epoch": args.n_epoch,
            "patience": args.patience}
    
    hyperparams_grid = []
    if os.path.isfile(args.hyperparams_path):
        assert args.dev is not None, "Need a dev set for hyperparameter search"		
        hyperparams_grid = helpers.get_hyperparams(args.hyperparams_path, conf)
        print "[tuning hyperparameters from @ {}]".format(args.hyperparams_path)
        if args.res_path is not None:            
            fname, _ = os.path.splitext(args.res_path)            
            hyper_results_path = fname+"_"+os.path.basename(args.test)+"_hyper2.txt"
        else:
            hyper_results_path = None
        scorer = lambda y_true,y_hat: f1_score(y_true, y_hat,average="macro") 	
        
    if args.cv is None:
        if len(hyperparams_grid) > 0:				            
            conf, _ = hypertuner(args.train, args.dev, args.test, args.emb_path,\
                                    scorer, hyperparams_grid, args.run_id, res_path=args.res_path)
        # main(args.train, args.dev, args.test, args.emb_path, conf, run_id=args.run_id, res_path=args.res_path)
    else:
        raise NotImplementedError
        assert args.cv > 2, "need at leat 2 folds for cross-validation"
        results = []
        cv_results_path = None
        if args.res_path is not None:
            #in CV experiments save the results of each fold in an external file
            fname, _ = os.path.splitext(args.res_path)
            cv_results_path = fname+"_"+os.path.basename(args.test)+"_CV.txt"
        else:
            cv_results_path = None
        for cv_fold in xrange(1, args.cv+1):
            tr_fname  = args.train+"_"+str(cv_fold)
            dev_fname = args.dev+"_"+str(cv_fold)
            ts_fname  = args.test+"_"+str(cv_fold)
            if len(hyperparams_grid) > 0:				            	
            	conf, _ = hypertune(tr_fname, ts_fname, args.emb_path,\
                                    scorer, hyperparams_grid, res_path=hyper_results_path)            
            #run model with the best hyperparams
            res = main(tr_fname, dev_fname, ts_fname, args.emb_path, conf, run_id=args.run_id, res_path=cv_results_path)            
            results.append(res)
        
        accs = [res["acc"] for res in results ]
        f1s = [res["avgF1"] for res in results ]
        
        cv_res = {"acc_mean":round(np.mean(accs),3), \
                "acc_std":round(np.std(accs),3), \
                "avgF1_mean":round(np.mean(f1s),3), \
                "avgF1_std":round(np.std(f1s),3), \
                "model":"NLSE", \
                "dataset":os.path.basename(args.test), \
                "run_id":args.run_id}
        helpers.print_results(cv_res)
        #save the results of each run 
        if args.res_path is not None:
            cols = ["dataset", "run_id", "model", "acc_mean","acc_std","avgF1_mean","avgF1_std"]
            helpers.save_results(cv_res, args.res_path, sep="\t", columns=cols)
            