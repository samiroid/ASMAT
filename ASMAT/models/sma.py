class SMA(object):

	def __init__(self):
		raise NotImplementedError

	def fit(self, X, Y, confs):
		raise NotImplementedError

	def predict_one(self, x):
		raise NotImplementedError

	def predict_many(self, X):
		raise NotImplementedError

	def predict_text_one(self, x):
		raise NotImplementedError

	def predict_text_many(self, X):
		raise NotImplementedError