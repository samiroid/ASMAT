import argparse
import cPickle
from ipdb import set_trace
import numpy as np
import os
import streaming_pickle as stPickle
import sys
import time

TRAIN_IDX = 1
NEG_SAMPLES_IDX = 4

def multinomial_samples(unigram_distribution, exclude=[], n_samples=1):    
    samples = []        
    while len(samples) != n_samples:            
        wrd_idx = np.argmax(np.random.multinomial(1, unigram_distribution))
        # from ipdb import set_trace; set_trace()
        if wrd_idx not in exclude: 
            samples.append(wrd_idx)                        
    return samples

def extract(instance, unigram_distribution, num_neg_samples):        
    train = instance[TRAIN_IDX]    
    neg_samples = []    
    for msg in train:            
        set_trace()
        neg_samp = [ multinomial_samples(unigram_distribution, msg, num_neg_samples) \
                     for _ in xrange(len(msg)) ]
        neg_samples.append(neg_samp)        
    instance[NEG_SAMPLES_IDX] = neg_samples    
    return instance

class negative_sampler():

    def __init__(self, word_count, index2word, warp=0.75):
        '''
        Store count for the range of indices in the dictionary
        '''
        max_index = max(index2word.keys())
        counts = []
        for n in range(max_index):
            if n in index2word:
                counts.append(word_count[index2word[n]]**warp)
            else:    
                counts.append(0)
        counts = np.array(counts)
        norm_counts = counts/sum(counts)
        scaling = int(np.ceil(1./min(norm_counts[norm_counts>0])))
        scaled_counts = (norm_counts*scaling).astype(int)
        self.cumsum = scaled_counts.cumsum()

    def sample(self, size=1):
        total_size = np.prod(size)
        random_ints = np.random.randint(self.cumsum[-1], size=total_size)
        data_y_neg = np.searchsorted(self.cumsum, random_ints).astype('int32')
        return data_y_neg.reshape(size) 

    def sample_2(self, exclude, size=1):
        total_size = np.prod(size)
        random_ints = np.random.randint(self.cumsum[-1], size=total_size)
        data_y_neg = np.searchsorted(self.cumsum, random_ints).astype('int32')
        #filter out words that should be excluded 
        filtered = [x for x in data_y_neg.tolist() if x not in exclude][:total_size]
        data_y_neg = np.array(filtered)
        return data_y_neg.reshape(size) 

def get_parser():
    parser = argparse.ArgumentParser(description="Extract negative samples")
    parser.add_argument('-input', type=str, required=True, help='train file')
    parser.add_argument('-aux_data', type=str, required=True, help='aux data file')
    parser.add_argument('-n_workers', type=int, help='number of jobs', default=1)    
    parser.add_argument('-neg_samples', type=int, help='number of negative samples', default=10)
    parser.add_argument('-resume', action="store_true", help='continue from a previous run', default=False)    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()      
    print "[training data: %s | aux_data: %s | n_workers: %d | resume: %s]" % (args.input, args.aux_data, args.n_workers, args.resume)             
    with open(args.aux_data,"r") as fid:
        wrd2idx, unigram_distribution, word_counter, _ = cPickle.load(fid)        

    index2word = {i:w for w,i in wrd2idx.items()}    
    sampler = negative_sampler(word_counter, index2word)

    t0 = time.time()
    prev_time = time.time()        
    tdf = open(args.input,"r")
    training_data = stPickle.s_load(tdf)        
    tmp_data_path = args.input.strip(".pkl")+"_new.pkl"
    new_training_data = open(tmp_data_path,"w")    

    # for instance in training_data:        
    #     neg_samples = []
    #     for msg in instance[TRAIN_IDX]:            
    #         neg_samples += [[ multinomial_samples(unigram_distribution, msg, args.neg_samples) \
    #                  for _ in xrange(len(msg)) ]]
    #     instance[NEG_SAMPLES_IDX] = neg_samples                  
    #     stPickle.s_dump_elt(instance,new_training_data)   

    for instance in training_data:        
        neg_samples = []
        for msg in instance[TRAIN_IDX]:            
            neg_samples += [sampler.sample((len(msg),args.neg_samples))]
        instance[NEG_SAMPLES_IDX] = neg_samples                  
        stPickle.s_dump_elt(instance,new_training_data)   

    #replace the training data file with the new augmented one    
    os.remove(args.input)
    os.rename(tmp_data_path, args.input)
    tend = time.time() - t0
    print "\n[runtime: %d minutes (%.2f secs)]" % ((tend/60),tend)    
