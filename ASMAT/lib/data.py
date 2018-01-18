import codecs
from collections import defaultdict
import numpy as np
import os
from ipdb import set_trace

def flatten_list(l):
    return [item for sublist in l for item in sublist]    

def shuffle_split(data, split_perc=0.8, random_seed=1234):
    """
        Split the data into train and test, keeping the class proportions

        data: list of (y,x) tuples
        split_perc: percentage of training examples in train/test split
        random_seed: ensure repeatable shuffles

        returns: balanced training and test sets
    """
    #shuffle data    
    rng = np.random.RandomState(random_seed)
    rng.shuffle(data)
    #group examples by class label
    z = defaultdict(list)
    for y, x in data: z[y].append(x)
    train = []
    test = []
    for label in z.keys():
        #examples of each label
        x_label = z[label]
        split = int(len(x_label) * split_perc)
        #train split        
        train_Xs = x_label[:split]
        train_Ys = [label] * len(train_Xs)        
        train += zip(train_Ys, train_Xs)
        #test split
        test_Xs = x_label[split:]
        test_Ys = [label] * len(test_Xs)        
        test += zip(test_Ys, test_Xs)
    #reshuffle
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test

def shuffle_split_idx(Y, split_perc = 0.8, random_seed=1234):
    """
        Split the data into train and test, keeping the class proportions

        data: list of labels
        split_perc: percentage of training examples in train/test split
        random_seed: ensure repeatable shuffles

        returns: indices for balanced training and test sets 
    """
    #shuffle data
    rng=np.random.RandomState(random_seed)          
    #group examples by class class    
    z = defaultdict(list)
    for i,y in enumerate(Y): z[y].append(i)    
    train = []    
    test  = []
    for cl in z.keys():
        #indices of the examples of each class 
        idx_cl = z[cl]  
        split   = int(len(idx_cl)*split_perc)
        train += idx_cl[:split]         
        test  += idx_cl[split:]
    #reshuffle
    rng.shuffle(train)
    rng.shuffle(test)    

    return train, test

def stratified_sampling(data, n, random_seed=1234):
    """
        Get a sample of the data, keeping the class proportions

        data: list of (x,y) tuples
        n: number of samples
        random_seed: ensure repeatable shuffles

        returns: balanced sample
    """
    rng=np.random.RandomState(random_seed)          
    z = defaultdict(list)
    #shuffle data
    rng.shuffle(data)
    #group examples by class    
    z = defaultdict(list)    
    for x,y in data: z[y].append(x)    
    #compute class distribution
    class_dist = {}
    for cl, samples in z.items():
        class_dist[cl] = int((len(samples)*1./len(data)) * n)
    sample = []    
    
    for label in z.keys():
        #examples of each label 
        x_label  = z[label]            
        sample += zip(x_label[:class_dist[label]],
                    [label] * class_dist[label])             
    #reshuffle
    rng.shuffle(sample)
    return sample

def simple_split(data, split_perc=0.8, random_seed=1234):
    """
        Split the data into train and test
        data: list of (y,x) tuples
        split_perc: percentage of training examples in train/test split
        random_seed: ensure repeatable shuffles
        returns: train and test splits
    """
    #shuffle data
    rng = np.random.RandomState(random_seed)
    rng.shuffle(data)    
    train = []
    test = []
    split = int(len(data) * split_perc)
    #train split
    train_split = data[:split]
    test_split = data[split:]
    
    return train_split, test_split

def kfolds(n_folds, n_elements, val_set=False, shuffle=False, random_seed=1234):   
         
    if val_set: assert n_folds>2    
    X = np.arange(n_elements)
    if shuffle: 
        rng=np.random.RandomState(random_seed)      
        rng.shuffle(X)    
    X = X.tolist()
    slice_size = n_elements/n_folds
    slices =  [X[j*slice_size:(j+1)*slice_size] for j in xrange(n_folds)]
    #append the remaining elements to the last slice
    slices[-1] += X[n_folds*slice_size:]
    kf = []
    for i in xrange(len(slices)):
        train = slices[:]     
        test = train.pop(i)
        if val_set:
            #take one of the slices as the development set
            try:
                val = train.pop(i)
            except IndexError:
                val = train.pop(-1)                
            #flatten the list of lists
            #train = [item for sublist in train for item in sublist]
            train = flatten_list(train)
            kf.append([train,test,val])
        else:
            #train = [item for sublist in train for item in sublist]
            train = flatten_list(train)
            kf.append([train,test])
    return kf

def crossfolds(data, k):    
    data = np.array(data)
    folds = []
    for i, (train, test) in enumerate(kfolds(k, len(data),shuffle=True)):
        train_data = data[train].tolist()
        test_data  = data[test].tolist() 
        folds.append([train_data, test_data])   
    return folds
    
def read_dataset(path, labels=None):	
    data = []
    ys = []
    with codecs.open(path, "r", "utf-8") as fid:
        for l in fid:
            splt = l.replace("\n", "").split("\t")
            y = splt[0]
            x = ' '.join(splt[1:])
            data.append([y,x])	
            ys+=[y]
    if labels is not None:
        data = filter_labels(data, labels)    
    return data

# def save_dataset(data, out_path, labels=None):
#     dirname = os.path.dirname(out_path)
#     if not os.path.exists(dirname):
#         os.makedirs(dirname)
#     if labels is not None: data = filter_labels(data, labels)
#     with open(out_path, "w") as fod:
#         for ex in data: fod.write('\t'.join(ex) + "\n")
#     return data

def save_dataset(data, out_path, labels=None):
    dirname = os.path.dirname(out_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if labels is not None: data = filter_labels(data, labels)
    with codecs.open(out_path, "w","utf-8") as fod:
        for ex in data: fod.write('\t'.join(ex) + "\n")
    return data


def filter_labels(data,labels):
    #dictionaries are faster for comparisons
    labels = {l:None for l in labels}
    filtered_data = filter(lambda x:x[0] in labels, data)
    return filtered_data

# def read_lexicon(path, sep='\t', ignore_above=float('inf'), ignore_below=-float('inf')):
#     with open(path) as fid:
#         lex = {wrd: float(scr) for wrd, scr in (line.split(sep) for line in fid)
#                if float(scr) < ignore_above and float(scr) > ignore_below}
#     return lex

def filter_lexicon(lexicon, ignore_above=float('inf'), ignore_below=-float('inf')):
    lex = { wrd: score for wrd, score in lexicon.items()
            if  float(score) < ignore_above 
            and float(score) > ignore_below }
    return lex

def save_lexicon(lexicon, path, sep='\t'):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path,"w") as fid:
        for word, score in lexicon.items():
            fid.write("{}{}{}\n".format(word, sep, float(score)))

def normalize_scores(lexicon, to_range=(0,1)):
    scores = lexicon.values()
    old_range = (min(scores),max(scores))
    for k in lexicon.keys():
        lexicon[k] = linear_conversion(old_range,to_range,lexicon[k])
    return lexicon

def linear_conversion(source_range, dest_range, val):
    MIN = 0
    MAX = 1
    val = float(val)
    source_range = np.asarray(source_range,dtype=float)
    dest_range = np.asarray(dest_range,dtype=float)
    new_value = ( (val - source_range[MIN]) / (source_range[MAX] - source_range[MIN]) ) *\
                (dest_range[MAX] - dest_range[MIN]) + dest_range[MIN]
    return round(new_value,3)

def read_lexicon(path, sep='\t', normalize=None):
    lex = None
    with open(path) as fid:
        lex = {wrd: float(scr) for wrd, scr in (line.split(sep) for line in fid)}
    if normalize is not None:
        assert isinstance(normalize, list) and \
        len(normalize) == 2, "please provide a range for normalization. e.g., [-1,1]"
        lex = normalize_scores(lex,normalize)
    return lex

    
