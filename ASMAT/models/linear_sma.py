from sma import SMA
import numpy as np
from collections import defaultdict
from sklearn.linear_model import SGDClassifier 

class Linear_SMA(SMA):

	def __init__(self):
		"""

		Parameters
		----------
		"""	
		self.model = SGDClassifier()
		
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
		raise NotImplementedError


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
		raise NotImplementedError

	def predict_one(self, x):
		return self.model.predict(x)

	def predict_many(self, X):
		return self.model.predict(X)

	def fit(self, X, Y, confs):
		self.model.fit(X,Y)




		