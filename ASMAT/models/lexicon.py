import argparse
import cPickle
from collections import defaultdict
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
from ASMAT.lib.data import read_dataset, read_lexicon, filter_lexicon
from ASMAT.lib import helpers

class LexiconSentiment(object):

	def __init__(self, lexicon=None, path=None, sep='\t', default_word_score=None,				
				ignore_score_above=float('inf'), ignore_score_below=-float('inf'),
				positive_threshold=0.5, negative_threshold=None, default_class=1,
				norm_scores=False, agg="mean",binary=False):
				
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
		lex = filter_lexicon(lex, ignore_above=ignore_score_above, 
								  ignore_below=ignore_score_below)
		#create a default dict to avoid KeyErrors
		try:
			default_word_score = float(default_word_score)
		except:
			default_word_score = None
		self.lexicon = defaultdict(lambda:default_word_score,lex)
		self.positive_threshold = positive_threshold
		self.negative_threshold = negative_threshold
		self.default_label = default_class	
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

