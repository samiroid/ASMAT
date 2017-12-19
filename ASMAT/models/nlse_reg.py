'''
SemEval models
'''

import cPickle
import numpy as np
import os
from pdb import set_trace
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
# DEBUGGING

def dropout(x, p, training=True, seed=1234):
    srng = RandomStreams(seed)
    if training:
        y = T.switch(srng.binomial(size=x.shape, p=p), x, 0)
    else:
        y = p*x 
    return y 

def init_W(size, rng, init=None, shared=True):
    '''
    Random initialization
    '''
    if len(size) == 2:
        n_out, n_in = size
    else:
        n_out, n_in = size[0], size[3]
    # Uniform init scaling
    if init == 'glorot-tanh':    
        w0 = np.sqrt(6./(n_in + n_out))   
    elif init == 'glorot-sigmoid':    
        w0 = 4*np.sqrt(6./(n_in + n_out))   
    else:
        w0 = init
    W = np.asarray(rng.uniform(low=-w0, high=w0, size=size))

    if shared:
        return theano.shared(W.astype(theano.config.floatX), borrow=True)
    else:
        return W.astype(theano.config.floatX)

class NN_reg():
    '''
    Embedding subspace
    '''
    def __init__(self, E, sub_size, model_file=None, init=None):
        # Random Seed
        if init is None:
            rng = np.random.RandomState(1234)        
        else:
            rng = np.random.RandomState(init)        
        if model_file:
            # Check conflicting parameters given 
            # if emb_path is not None or sub_size is not None:
            #     raise EnvironmentError, ("When loading a model emb_path and "
            #                              "sub_size must be set to None")
            # Load pre existing model  
            with open(model_file, 'rb') as fid: 
                [E, S, C, emb_path] = cPickle.load(fid)
            E             = theano.shared(E, borrow=True)
            S             = theano.shared(S, borrow=True)
            C             = theano.shared(C, borrow=True)
            # self.emb_path = emb_path
        else:
            # Embeddings e.g. word2vec.   
            # with open(emb_path, 'rb') as fid:
            #     E = cPickle.load(fid).astype(theano.config.floatX)
            emb_size, voc_size = E.shape
            # This is fixed!
            E = theano.shared(E, borrow=True)
            # Embedding subspace projection
            S = init_W((sub_size, emb_size), rng, init=0.1) # 0.0991
            # Hidden layer
            C = init_W((3, sub_size), rng, init=0.7) # 0.679
            # Store the embedding path used
            # self.emb_path = emb_path

        # Fixed parameters
        self.params = [E, S, C]
        # Compile
        self.compile()

    def forward(self, x):
        return self.fwd(x.astype('int32'))

    def compile(self):
        '''
        Forward pass and Gradients
        '''        
        E, S, C = self.params
        # FORWARD PASS
        # Embedding layer subspace        
        self.z0 = T.ivector()                            
        z1 = T.nnet.sigmoid(E[:, self.z0])  # embedding
        z2 = T.nnet.sigmoid(T.dot(S, z1))  # subspace
        # Hidden layer
        self.hat_y = T.dot(C, z2)[0]
        # Compile forward pass
        self.fwd = theano.function(
            [self.z0], self.hat_y, allow_input_downcast=True)
        # TRAINING COST        
        self.z1 = z1        
        self.y = T.fscalar()
        #squared loss error
        self.F = ((self.hat_y - self.y)**2).sum()
        self.cost = theano.function([self.z0, self.y], self.F)
        # Naming
        self.z0.name = 'z0'
        self.z1.name = 'z1'
        self.y.name = 'y'
        self.F.name = 'F'

    def save(self, model_file):
        #create output folder if needed
        if not os.path.exists(os.path.dirname(model_file)):
            os.makedirs(os.path.dirname(model_file))
        with open(model_file, 'wb') as fid: 
            param_list = [W.get_value() for W in self.params] #+ [self.emb_path]
            cPickle.dump(param_list, fid, cPickle.HIGHEST_PROTOCOL)

    def load(self, path):
        # Load pre existing model  
        with open(path, 'rb') as fid: 
            E, S, C = cPickle.load(fid)
            E = theano.shared(E, borrow=True)
            S = theano.shared(S, borrow=True)
            C = theano.shared(C, borrow=True)
        self.params = [E, S, C]








