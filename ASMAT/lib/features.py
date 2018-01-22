import numpy as np
import os


def NLSE(X, Y):
	lens = np.array([len(tr) for tr in X]).astype(int)
	st = np.cumsum(np.concatenate((np.zeros((1, )), lens[:-1]), 0)).astype(int)
	ed = (st + lens)
	x = np.zeros((ed[-1], 1))
	for i, ins_x in enumerate(X):
		x[st[i]:ed[i]] = np.array(ins_x, dtype=int)[:, None]
	X = x
	Y = np.array(Y)

	return X, Y, st, ed

def BOW(docs, wrd2idx, agg='freq'):
	"""
		Extract bag-of-word features
	"""
	assert agg in ["freq", "bin"]
	X = np.zeros((len(docs), len(wrd2idx)))
	for i, doc in enumerate(docs):
		try:
			X[i, np.array(doc)] = 1
		except IndexError:
			pass
	return X


def BOE(docs, E, agg='sum', mean=False):
	"""
		Build Bag-of-Embedding features
	"""
	assert agg in ["sum", "bin"]
	X = np.zeros((len(docs), E.shape[0]))
	if agg == 'sum':
		for i, doc in enumerate(docs):
			X[i, :] = E[:, doc].T.sum(axis=0)
	elif agg == 'bin':
		for i, doc in enumerate(docs):
			unique = list(set(doc))
			X[i, :] = E[:, unique].T.sum(axis=0)
	return X

# def Brown_Clusters(brown_clusters, user_2_words, wrd2idx, depth=None):
# 	cluster2idx = {c:i for i,c in enumerate(set(brown_clusters.values()))}
# 	idx2wrd = {i:w for w,i in wrd2idx.items()}
# 	n_clusters = len(set(brown_clusters.values()))
# 	X = np.zeros((len(user_2_words),n_clusters))
# 	for x, doc in user_2_words.items():
# 		clusters = [ brown_clusters[idx2wrd[w]] for w in doc if w in brown_clusters]
# 		cluster_ids = map(lambda x:cluster2idx[x], clusters)
# 		X[x,cluster_ids] = 1
# 	return X

# def lda_features(lda_path, lda_idx_path, user_2_words, wrd2idx):

# 	lda_reader = LDAReader(None)
# 	lda_reader.load_model(lda_path, lda_idx_path)
# 	#note that gensim needs the actual documents so I am converting from indices back to actual words
# 	idx2wrd = {i:w for w,i in wrd2idx.items()}

# 	# docs = [" ".join([idx2wrd[w] for w in m ]) for m in X_words]
# 	# X_lda = np.array([lda_reader.get_topics(doc, binary=True) for doc in docs])

# 	X = np.zeros((len(user_2_words),lda_reader.model.num_topics))

# 	for x, words in user_2_words.items():
# 		txt = " ".join([idx2wrd[w] for w in words ])
# 		topic_vector = lda_reader.get_topics(txt, binary=True)
# 		# print "*"*90
# 		# print topic_vector
# 		X[x,:] = topic_vector
# 		# print "*"*90
# 	return X
