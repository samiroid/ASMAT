import cPickle
import numpy as np
import os
try:
    from ipdb import set_trace
except ImportError:
    from pdb import set_trace
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.metrics import f1_score, accuracy_score
import sys
import uuid
RAND_SEED=1234

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
        w0 = 1
    W = np.asarray(rng.uniform(low=-w0, high=w0, size=size))
    if shared:
        return theano.shared(W.astype(theano.config.floatX), borrow=True)
    else:
        return W.astype(theano.config.floatX)

def build_input(X):
	lens = np.array([len(tr) for tr in X]).astype(int)
	st = np.cumsum(np.concatenate((np.zeros((1, )), lens[:-1]), 0)).astype(int)
	ed = (st + lens)
	x = np.zeros((ed[-1], 1))
	for i, ins_x in enumerate(X):
		x[st[i]:ed[i]] = np.array(ins_x, dtype=int)[:, None]
	X = x
	return X, st, ed

def colstr(string, color):
    if color is None:
        cstring = string
    elif color == 'red':
        cstring = "\033[31m" + string  + "\033[0m"
    elif color == 'green':    
        cstring = "\033[32m" + string  + "\033[0m"
    return cstring    

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

def evaluate(model, X, Y):
    # Evaluation
    Y_hat = model.predict(X)
    avgF1 = f1_score(Y, Y_hat,average="macro")        
    acc = accuracy_score(Y, Y_hat)            
    return avgF1, acc

def train_model(nn, train_x, train_y, dev_x, dev_y, silent=False):    
    TMP_MODELS="/tmp/subspace_"+str(uuid.uuid4())+".pkl"
    print "[temp model: {}]".format(TMP_MODELS)
    train_x, st, ed = build_input(train_x)
    n_sent_train = len(st)    
    train_y = np.array(train_y)    
    # Ensure types compatible with GPU
    train_x = train_x.astype('int32')
    # Otherwise slices are scalars not Tensors
    train_y = train_y[:, None].astype('int32')
    st = st.astype('int32')
    ed = ed.astype('int32')
    # Store as shared variables (push into the GPU)
    train_x = theano.shared(train_x, borrow=True)
    train_y = theano.shared(train_y, borrow=True)
    st = theano.shared(st, borrow=True)
    ed = theano.shared(ed, borrow=True)
    # SGD Update rule    
    print "[update embeddings: {}]".format(nn.tune_embeddings)
    if nn.tune_embeddings:        
        updates = [(pr, pr - nn.lrate * T.grad(nn.F, pr))
               for pr in nn.params]
    else:
        # Sub-space: Do not update E
        updates = [(pr, pr - nn.lrate * T.grad(nn.F, pr))
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
    drops=0
    
    for i in np.arange(nn.n_epoch):
        # Epoch train
        obj = 0
        n = 0
        if nn.randomize:
            nn.rng.shuffle(train_idx)
        for j in train_idx:
            obj += train_batch(j)
            # INFO
            if not n % 10 and not silent:
                print "\rEpoch: %d\%d | %d/%d %s" % (i + 1, nn.n_epoch, n + 1, n_sent_train, " "),
                sys.stdout.flush()
            n += 1
        Fm, cr = evaluate(nn, dev_x, dev_y)
        # INFO
        if last_Fm:
            if best_Fm[0] < Fm:
                # Keep best model
                best_Fm = [Fm, i + 1]
                nn.save(TMP_MODELS)
                best = '*'
                drops=0
            else:
                best = ''
                drops+=1
            delta_Fm = Fm - last_Fm
            if delta_Fm >= 0:
                delta_str = colstr("+%2.2f" % (delta_Fm * 100), 'green')
            else:
                delta_str = colstr("%2.2f" % (delta_Fm * 100), 'red')
                # if nn.exp_decay:
                #     nn.lrate*=10**10
                    # nn.lrate/=2
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
            nn.save(TMP_MODELS)
        if last_Acc:
            if last_Acc > cr:
                acc_str = "Acc " + colstr("%2.2f%%" % (cr * 100), 'red')
            else:
                acc_str = "Acc " + colstr("%2.2f%%" % (cr * 100), 'green')
        else:
            acc_str = "Acc %2.2f%%" % (cr * 100)
        last_Acc = cr
        last_Fm = Fm
        items = (i + 1, nn.n_epoch, obj_str,
                 acc_str, Fm * 100, delta_str, best)        
        if not silent:            
            print "%s Fm %2.2f%% %s%s" % items[3:]
        if drops >= nn.patience:
            print "[ran out of patience]"
            break
    #load best model
    nn.load(TMP_MODELS)
    return nn

def load_model(model_path):
    clf = NLSE(None,None,None,None,None)
    clf.load(model_path)
    return clf

class NLSE(object):
    '''
    Embedding subspace
    '''
    
    def __init__(self, E, sub_size, label_map, vocab=None, lrate=0.01, 
                tune_embeddings=False, n_epoch=10, randomize_train=True, 
                weight_CM=None, rand_seed=1234, init='glorot-tanh', patience=10,
                early_bow=False):
        # Random Seed
        self.rng = np.random.RandomState(rand_seed)    
        self.lrate = lrate
        self.n_epoch = n_epoch
        self.randomize = randomize_train
        self.label_map = label_map 
        self.vocab = vocab                 
        self.patience = patience
        self.weight_CM = weight_CM
        self.tune_embeddings = tune_embeddings
        self.early_bow = early_bow
        if E is not None and sub_size is not None and label_map is not None:            
            # Embedding Layer
            emb_size = E.shape[0]
            E = E.astype(theano.config.floatX) 
            E = theano.shared(E, borrow=True)        
            # Embedding subspace projection        
            S = init_W((sub_size, emb_size), self.rng, init=init) # 0.0991
            # Hidden layer
            C = init_W((len(label_map), sub_size), self.rng, init=init) # 0.679        
            self.params = [E, S, C]
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
        self.z0    = T.ivector()                  # one hot
        z1         = E[:, self.z0]                # embedding
        z2         = T.nnet.sigmoid(T.dot(S, z1)) # subspace        
        if self.early_bow:
            z3         = T.sum(z2, 1)                 # Bag of words
            z4         = T.dot(C, z3)                 # Hidden layer            
        else:
            z3         = T.dot(C, z2)                 # Hidden layer
            z4         = T.sum(z3, 1)                 # Bag of words
        self.hat_y = T.nnet.softmax(z4.T).T
        # Compile forward pass
        self.fwd = theano.function([self.z0], self.hat_y)        
        # TRAINING COST 
        # Train cost minus log probability
        self.z1 = z1
        self.y  = T.ivector()                             
        #weighted confusion matrix (penalize errors differently)
        if self.weight_CM:
            WCM    = (self.weight_CM[self.y, :].T)*T.log(self.hat_y)
            self.F = -T.mean(WCM.sum(0))        
        else:
            self.F = -T.mean(T.log(self.hat_y)[self.y])        
        self.cost = theano.function([self.z0, self.y], self.F)

        # Naming
        self.z0.name = 'z0'
        self.z1.name = 'z1'
        self.y.name  = 'y'
        self.F.name  = 'F'

    def predict(self, X):
        Y_hat = np.zeros(len(X), dtype='int32')
        # dev_p_y = np.zeros((3, dev_y.shape[0]))
        for j, x in enumerate(X):
            # Prediction
            x = np.array(x)
            p_y = self.forward(x)
            hat_y = np.argmax(p_y)
            Y_hat[j]=hat_y
        return Y_hat
    
    def fit(self, train_x, train_y, dev_x, dev_y, silent=False):
        nn = train_model(self,train_x, train_y, dev_x, dev_y, silent)
        self.params = nn.params

    def save(self, model_file):
        #create output folder if needed
        if not os.path.exists(os.path.dirname(model_file)):
            os.makedirs(os.path.dirname(model_file))
        with open(model_file, 'wb') as fid: 
            param_list = [W.get_value() for W in self.params] #+ [self.emb_path]
            param_list += [self.vocab, self.label_map, self.weight_CM]
            cPickle.dump(param_list, fid, cPickle.HIGHEST_PROTOCOL)

    def load(self, path):
        # Load pre existing model  
        with open(path, 'rb') as fid: 
            E, S, C, vocab, label_map, weight_CM = cPickle.load(fid)
            E = theano.shared(E, borrow=True)
            S = theano.shared(S, borrow=True)
            C = theano.shared(C, borrow=True)
        self.params = [E, S, C]
        self.vocab = vocab
        self.label_map = label_map
        self.weight_CM = weight_CM
        self.compile()



class BOE_plus(object):
    '''
     BOE Plus
    '''    
    def __init__(self, E, label_map, vocab=None, lrate=0.01, 
                tune_embeddings=False, n_epoch=10, randomize_train=True, 
                weight_CM=None, rand_seed=1234, init='glorot-tanh', patience=10):
        # Random Seed
        self.rng = np.random.RandomState(rand_seed)    
        self.lrate = lrate
        self.n_epoch = n_epoch
        self.randomize = randomize_train
        self.label_map = label_map 
        self.vocab = vocab                 
        self.patience = patience
        self.weight_CM = weight_CM
        self.tune_embeddings = tune_embeddings
        if E is not None and label_map is not None:            
            # Embedding Layer
            emb_size = E.shape[0]
            E = E.astype(theano.config.floatX) 
            E = theano.shared(E, borrow=True)                    
            # Prediction layer
            C = init_W((len(label_map), emb_size), self.rng, init=init) # 0.679        
            self.params = [E, C]
            self.compile()        

    def forward(self, x):
        return self.fwd(x.astype('int32'))

    def compile(self):
        '''
        Forward pass and Gradients
        '''        
        E, C = self.params
        # FORWARD PASS
        # Embedding layer subspace
        self.z0    = T.ivector()                  # one hot
        z1         = E[:, self.z0]                # embedding        
        z2         = T.sum(z1, 1)                 # Bag of words
        z3         = T.dot(C, z2)                 # Prediction layer        
        self.hat_y = T.nnet.softmax(z3.T).T
        # Compile forward pass
        self.fwd = theano.function([self.z0], self.hat_y)        
        # TRAINING COST 
        # Train cost minus log probability
        self.z1 = z1
        self.y  = T.ivector()                             
        #weighted confusion matrix (penalize errors differently)
        if self.weight_CM:
            WCM    = (self.weight_CM[self.y, :].T)*T.log(self.hat_y)
            self.F = -T.mean(WCM.sum(0))        
        else:
            self.F = -T.mean(T.log(self.hat_y)[self.y])        
        self.cost = theano.function([self.z0, self.y], self.F)

        # Naming
        self.z0.name = 'z0'
        self.z1.name = 'z1'
        self.y.name  = 'y'
        self.F.name  = 'F'

    def predict(self, X):
        Y_hat = np.zeros(len(X), dtype='int32')
        # dev_p_y = np.zeros((3, dev_y.shape[0]))
        for j, x in enumerate(X):
            # Prediction
            x = np.array(x)
            p_y = self.forward(x)
            hat_y = np.argmax(p_y)
            Y_hat[j]=hat_y
        return Y_hat
    
    def fit(self, train_x, train_y, dev_x, dev_y, silent=False):
        nn = train_model(self,train_x, train_y, dev_x, dev_y, silent)
        self.params = nn.params

    def save(self, model_file):
        #create output folder if needed
        if not os.path.exists(os.path.dirname(model_file)):
            os.makedirs(os.path.dirname(model_file))
        with open(model_file, 'wb') as fid: 
            param_list = [W.get_value() for W in self.params] #+ [self.emb_path]
            param_list += [self.vocab, self.label_map, self.weight_CM]
            cPickle.dump(param_list, fid, cPickle.HIGHEST_PROTOCOL)

    def load(self, path):
        # Load pre existing model  
        with open(path, 'rb') as fid: 
            E, C, vocab, label_map, weight_CM = cPickle.load(fid)
            E = theano.shared(E, borrow=True)            
            C = theano.shared(C, borrow=True)
        self.params = [E, C]
        self.vocab = vocab
        self.label_map = label_map
        self.weight_CM = weight_CM
        self.compile()

