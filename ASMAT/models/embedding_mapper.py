import numpy as np
import theano 
import theano.tensor as T
import cPickle 

class Emb_Mapper():

    def __init__(self, Emb_in, Emb_out,lrate=0.01):
        #initializations
        rng = np.random.RandomState(1234)        
        I = theano.shared(Emb_in.astype(theano.config.floatX))
        O = theano.shared(Emb_out.astype(theano.config.floatX))
        self.W = self.init_W((Emb_out.shape[1],Emb_in.shape[1]), rng)               
        #model
        # from ipdb import set_trace; set_trace()
        x = T.iscalar('x')
        x_in = I[x,:]
        x_out = O[x,:]
        hat_x_out = T.dot(self.W,x_in)
        diff = hat_x_out - x_out
        #cost
        J = (diff ** 2).sum()
     
        grad_W = T.grad(J,self.W) 
        updates = ((self.W, self.W - lrate * grad_W),)
        self.train = theano.function(inputs=[x],                                
                                      outputs=J,
                                      updates=updates)
    
    def init_W(self, size, rng):                
        W = np.asarray(rng.uniform(low=-1, high=1, size=size))
        return theano.shared(W.astype(theano.config.floatX), borrow=True)

    def save(self, path):
        with open(path,"wb") as fod:
            cPickle.dump(self.W.get_value(), fod, cPickle.HIGHEST_PROTOCOL)