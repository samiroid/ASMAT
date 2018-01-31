import argparse
import cPickle
from collections import defaultdict, Counter
from ipdb import set_trace
import pprint
import json
import math
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sys
sys.path.append("..")
from ASMAT.lib.data import read_dataset
from ASMAT.lib import helpers

def filter_lexicon(lexicon, keep_below, keep_above):
    lex = { wrd: score for wrd, score in lexicon.items()
            if  float(score) <= keep_below
            or float(score) >= keep_above }
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

class LexiconSentiment(object):

	def __init__(self, lexicon=None, path=None, sep='\t', default_word_score=None,				
				keep_scores_below=None, keep_scores_above=None,
				positive_threshold=0.5, negative_threshold=None, 
				default_label=1,norm_scores=True, agg="mean",binary=False):
				
		"""
		Parameters
		----------
		lexicon: dict
				lexixon
		path: string
			 path to a lexicon
		sep: string
			separator between words and scores (when loading lexicon from file)
		ignore_scores: [float, flot]
			ignore words with scores between these scores 		
		binary: only considers each word once 
		"""
		assert path is not None or lexicon is not None, \
			"Must pass either a lexicon or a path to a lexicon"
		assert not (path is not None and lexicon is not None), \
			"Must pass a lexicon or a path to a lexicon (but not both!)"
		
		#load lexicon
		if path is not None:
			if norm_scores:				
				lex = read_lexicon(path,sep=sep,normalize=[-1,1])				
			else:
				lex = read_lexicon(path, sep=sep)
		else:
			lex = lexicon
		
		assert isinstance(lex, dict), "lexicon must be a dictionary"		
		lex = filter_lexicon(lex, keep_below=keep_scores_below,
								    keep_above=keep_scores_above)		
		#create a default dict to avoid KeyErrors
		try:
			default_word_score = float(default_word_score)
		except:
			default_word_score = None
		
		self.norm_scores = norm_scores
		self.keep_scores_below = keep_scores_below
		self.keep_scores_above = keep_scores_above
		self.lexicon = defaultdict(lambda:default_word_score,lex)
		self.positive_threshold = positive_threshold
		self.negative_threshold = negative_threshold
		self.default_label = default_label	
		self.binary = binary
		if agg == "sum":
			self.agg = np.sum
		else:
			self.agg = np.mean		
			
	def __predict(self, doc, dbg=False):
		"""
			Make a prediction for a single document based on the lexicon
			This method assumes that the document can be correctly tokenized with white-spaces. 

			Parameters
			----------
			doc: string
				 input document. 

		 	Returns
			-------
			numpy.array
				scores inferred from the lexicon
				the output will be depend on the paramaters of the model: 
				``score_sum``, ``score_mean``, ``score_std``
		"""		
		if self.binary:
			word_scores = [self.lexicon[w] for w in list(set(doc.split()))] 		
		else:
			word_scores = [self.lexicon[w] for w in doc.split()] 		
		word_scores = [s for s in word_scores if s is not None]
		y_hat = self.default_label
		document_score = None
		#if no words from the lexicon are found, predict default class
		if len(word_scores) > 0:
			document_score = self.agg(word_scores)
			if document_score > self.positive_threshold:
				y_hat = 1
			elif document_score < self.positive_threshold:
				y_hat = -1
		if dbg:
			try:
				std = round(np.std(word_scores),3)
			except TypeError:
				std = None			
			try:
				document_score = round(document_score,3)
			except TypeError:
				pass
			return [y_hat, document_score, std, word_scores]
		return y_hat

	def predict(self, X):		
		if isinstance(X, unicode):
			return self.__predict(X)
		elif isinstance(X, list):
			predictions = [self.__predict(x) for x in X]
			return np.array(predictions)
		else:
			raise AssertionError("input must be either a string or a list of strings")

	def debug(self, X):
		predictions = [self.__predict(x, dbg=True) for x in X]		
		return predictions
	
	def fit(self, X, Y,  samples=20, silent=False):
		"""
			get optimal threshold from a labeled dataset by computing the mean and std 
			and then grid searching in a window of std+k around the mean (k is a constant)			
		"""
		# set_trace()
		predictions = [self.__predict(x, dbg=True)[1] for x in X]
		positive_idx = np.where(np.array(Y) == 1)[0]
		negative_idx = np.where(np.array(Y) == -1)[0]
		# set_trace()
		positives =  [predictions[i] for i in positive_idx if predictions[i] is not None]
		negatives =  [predictions[i] for i in negative_idx if predictions[i] is not None]
		
		#remove Nones
		predictions = [x for x in predictions if x is not None]
		mu_positive = round(np.mean(positives),3)
		mu_negative = round(np.mean(negatives),3)
		sig_positive = round(np.std(positives),3)				
		sig_negative = round(np.std(negatives),3)				
		lower = round(mu_negative - sig_negative,3)
		upper = round(mu_positive + sig_positive,3)
		# print "[search:{}]".format(repr([lower,upper]))
		thresholds = [round(n,3) for n in np.linspace(lower,upper,samples)]
		# thresholds = [round(n,3) for n in np.linspace(mu-(sig+c),mu+(sig+c),samples)]
		best_t = self.positive_threshold
		self.default_label = Counter(Y).most_common(1)[0][0]
		# set_trace()
		best_score = 0
		for t in thresholds:
			self.positive_threshold = t
			Y_hat = self.predict(X)					
			score = f1_score(Y, Y_hat,average="macro") 	
			if score > best_score:
				if not silent: print "[best >> t:{} | score:{}]".format(t,score)
				best_score = score
				best_t = t
		self.positive_threshold = best_t
	
	def get_params(self):
		return {"norm_scores":self.norm_scores,
				"keep_scores_below": self.keep_scores_below, 
				"keep_scores_above": self.keep_scores_above, 		
				"positive_threshold": self.positive_threshold, 
				"negative_threshold": self.negative_threshold, 
				"default_label": self.default_label, 	
				"binary": self.binary 
				}

	# def fit_old(self, X, Y, c=0.1, samples=20, silent=False):
	# 	"""
	# 		get optimal threshold from a labeled dataset by computing the mean and std 
	# 		and then grid searching in a window of std+k around the mean (k is a constant)			
	# 	"""
	# 	predictions = [self.__predict(x, dbg=True)[1] for x in X]
	# 	#remove Nones
	# 	predictions = [x for x in predictions if x is not None]
	# 	mu = round(np.mean(predictions),3)
	# 	sig = round(np.std(predictions),3)	
	# 	lower = round(mu-(sig+c),3)
	# 	upper = round(mu+(sig+c),3)
	# 	print "[search:{}]".format(repr([lower,upper]))			
	# 	thresholds = [round(n,3) for n in np.linspace(lower,upper,samples)]
	# 	best_t = self.positive_threshold
	# 	self.default_label = Counter(Y).most_common(1)[0][0]
	# 	# set_trace()
	# 	best_score = 0
	# 	for t in thresholds:
	# 		self.positive_threshold = t
	# 		Y_hat = self.predict(X)					
	# 		score = f1_score(Y, Y_hat,average="macro") 	
	# 		if score > best_score:
	# 			if not silent: print "[best >> t:{} | score:{}]".format(t,score)
	# 			best_score = score
	# 			best_t = t
	# 	self.positive_threshold = best_t

class LexiconLogOdds(LexiconSentiment):

	def __init__(self, lexicon=None, path=None, sep='\t', 
				ignore_score_above=float('inf'), ignore_score_below=-float('inf'),
				default_class=1, binary=True, threshold=0.5):
		#parent constructor
		LexiconSentiment.__init__(self, lexicon, path, sep, default_word_score=0, 
								 norm_scores=False, default_class=default_class,ignore_score_above=ignore_score_above, 
								 ignore_score_below=ignore_score_below,
								 binary=binary)
		
		self.positive_lexicon = self.lexicon
		self.negative_lexicon = {k:round(1 - v, 3) for k,v in self.lexicon.items()}
		self.negative_lexicon = defaultdict(lambda: 0, self.negative_lexicon)
		self.threshold = threshold
			
	def __predict(self, doc, dbg=False):
		"""
			Make a prediction for a single document based on the lexicon
			This method assumes that the document can be correctly tokenized with white-spaces. 

			Parameters
			----------
			doc: string
				 input document. 

		 	Returns
			-------
			numpy.array
				scores inferred from the lexicon
				the output will be depend on the paramaters of the model: 
				``score_sum``, ``score_mean``, ``score_std``
		"""		
		
		if self.binary:
			tokens = list(set(doc.split()))
		else:
			tokens = doc.split()
			
		positive_word_scores = [self.positive_lexicon[w] for w in tokens] 	
		negative_word_scores = [self.negative_lexicon[w] for w in tokens] 			
		positive_doc_score = np.sum(positive_word_scores)
		negative_doc_score = np.sum(negative_word_scores)
		positive_prob = positive_doc_score/(negative_doc_score+positive_doc_score)
		negative_prob = negative_doc_score/(negative_doc_score+positive_doc_score)
		# set_trace()
		
		if positive_prob == negative_prob:
			y_hat = self.default_label
		elif positive_prob > self.threshold:
			y_hat = 1
		else:
			y_hat = -1
		if dbg:
			return [y_hat, round(positive_doc_score,3), round(negative_doc_score,3), 
					round(positive_prob,3), round(negative_prob,3)]
		return y_hat

	def predict(self, X):		
		if isinstance(X, unicode):
			return self.__predict(X)
		elif isinstance(X, list):
			predictions = [self.__predict(x) for x in X]
			return np.array(predictions)
		else:
			raise AssertionError("input must be either a string or a list of strings")
	
	def debug(self, X):
		predictions = [self.__predict(x, dbg=True) for x in X]		
		return predictions

