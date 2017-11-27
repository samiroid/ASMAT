from sma import SMA
import numpy as np
from collections import defaultdict

class Lexicon_SMA(SMA):

	def __init__(self, lexicon=None, path=None, sep='\t', 
				 ignore_above=float('inf'), ignore_below=-float('inf'),
				 ignore_unks_on_mean=True, score_sum=False, score_mean=False,
				 score_std=False):
		"""

		Parameters
		----------
		lexicon: dict
				lexixon
		path: string
			 path to a lexicon
		sep: string
			separator between words and scores (when loading lexicon from file)
		ignore_above: float
			ignore words with scores above ``ignore_above``
		ignore_below: float
			ignore words with scores below ``ignore_below``
		ignore_unks_on_mean: bool
			 If True do include words not present in the lexicon when computing
			 the mean scores
		score_sum: bool
			 If True include the sum of scores on the predictions 			 
		score_mean: bool
			 If True include the mean of scores on the predictions 			 
		score_std: bool
			 If True include the standard deviation of scores on the preditctions 
		"""
		assert path is not None or lexicon is not None, \
			"Must pass either a lexicon or a path to a lexicon"
		assert not (path is not None and lexicon is not None), \
			"Must pass a lexicon or a path to a lexicon (but not both!)"
		assert score_sum or score_mean or score_std, "Must choose a score type"
		
		#load lexicon
		if lexicon is not None:
			assert type(lexicon) == dict, "lexicon must be a dictionary"
			lex = { wrd:score for wrd, score in lexicon.items() \
							if  float(score) < ignore_above \
							and float(score) > ignore_below }
		elif path is not None:
			with open(path) as fid:	
				lex =  { wrd: float(scr) for wrd, scr in (line.split(sep) \
								  for line in fid) \
				 			       if float(scr) < ignore_above \
								  and float(scr) > ignore_below }
		else:
			raise RuntimeError("this is ackward")
		#create a default dict to avoid KeyErrors
		self.lexicon = defaultdict(float)
		self.lexicon.update(lex)
		#other parameters
		self.ignore_unks_on_mean = ignore_unks_on_mean
		self.score_sum = score_sum
		self.score_mean = score_mean
		self.score_std = score_std

	def predict_text_one(self,doc):
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
		word_scores = map(lambda x:self.lexicon[x], doc.split())
		if self.ignore_unks_on_mean:
			word_scores = filter(lambda x:x!=0, word_scores)
		predictions = []
		if self.score_sum:  predictions.append(np.sum(word_scores))
		if self.score_mean: predictions.append(np.mean(word_scores))
		if self.score_std:  predictions.append(np.std(word_scores))
		return np.array(predictions)


	def predict_text_many(self, docs):
		
		"""
			Make predictions for a list of documents based on the lexicon
			This method assumes that the documents can be correctly tokenized with white-spaces. 

			Parameters
			----------
			docs: list
				 list of input documents.

			Returns
			-------
			numpy.array
				scores inferred from the lexicon
		"""
		predictions = [ self.predict_text_one(doc) for doc in docs ]
		return np.array(predictions)




		