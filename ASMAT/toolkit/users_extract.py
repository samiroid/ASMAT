import argparse
import codecs
import cPickle
from ipdb import set_trace
import os
import sys
sys.path.append("..")

from ASMAT.lib.vectorizer import docs2idx, build_vocabulary
from ASMAT.lib import embeddings
from ASMAT.lib.data import read_dataset, flatten_list, filter_labels

# def get_vocabulary(fnames, max_words=None):	
# 	datasets = []	
# 	for fname in fnames:		
# 		ds = read_dataset(fname)
# 		datasets.append(ds)
# 	vocab_docs = [x[1] for x in flatten_list(datasets)]	
# 	vocab = build_vocabulary(vocab_docs, max_words=max_words)
# 	return vocab

# def main(fnames, vocab, opts):
# 	#read data
# 	datasets = []
# 	for fname in fnames:
# 		print "[reading data @ {}]".format(repr(fname))
# 		ds = read_dataset(fname, labels=opts.labels)
# 		datasets.append(ds)	
# 	#vectorize
# 	print "[vectorizing documents]"
# 	for name, ds in zip(fnames, datasets):
# 		X, Y, label_map = vectorize(ds, vocab, opts.idx_labels)
# 		basename = os.path.splitext(os.path.basename(name))[0]
# 		path = opts.out_folder + basename
# 		print "[saving data @ {}]".format(path)
# 		with open(path, "wb") as fid:
# 			cPickle.dump([X, Y, vocab, label_map], fid, -1)
# 	return vocab


def get_vocabularies(text_paths, label_paths, max_words=None, users_only=False):	
	"""
		compute vocabulary using the texts from users only considering the users from a specific set of labeled datasets

		text_path: path to a file with all the text from a user in the format: USER TAB TEXT
		label_paths: list of paths to files with the labels in the format: LABEL USER

	"""	
	#read the labeled sets
	datasets = []
	for fname in label_paths:		
		ds = read_dataset(fname)				
		datasets.append(ds)
	vocab_datasets = [x[1] for x in flatten_list(datasets)]	
	words_vocab = None
	
	if not users_only:
		text_data = []
		for fname in text_paths:
			text_data += read_dataset(fname)	
		#index tweets per user
		text_by_user = {x[0]:x[1] for x in text_data}	
		vocab_docs = [text_by_user[x] for x in vocab_datasets]	
		words_vocab = build_vocabulary(vocab_docs, max_words=max_words)

	users_vocab = build_vocabulary([x for x in vocab_datasets])	
	return words_vocab, users_vocab

def vectorize(dataset, vocabulary):
	docs = [x[1] for x in dataset]
	Y = [x[0] for x in dataset]
	X, _ = docs2idx(docs, vocabulary)		
	return X, Y

def main(text_paths, label_paths, wrd2idx, user2idx, opts):	
	user_datasets = []
	for fname in label_paths:
		print "[reading user data @ {}]".format(repr(fname))
		ds_users = read_dataset(fname)		
		user_datasets.append(ds_users)
		# document_datasets.append(ds_docs)
	#vectorize
	print "[vectorizing users]"
	for name, ds in zip(label_paths, user_datasets):		
		X, Y = vectorize(ds, user2idx)
		basename = os.path.splitext(os.path.basename(name))[0]
		path = opts.out_folder + basename + "_users"
		print "[saving data @ {}]".format(path)
		with open(path, "wb") as fid:
			cPickle.dump([X, Y, user2idx], fid, -1)

	if not opts.users_only:		
		#read text data
		text_data = []
		for fname in text_paths:
			print "[reading text data @ {}]".format(repr(fname))
			text_data += read_dataset(fname)	
		#index tweets per user
		text_by_user = {x[0]:x[1] for x in text_data}			
		print "[vectorizing documents]"
		for name, ds in zip(label_paths, user_datasets):		
			ds_docs = [[y,text_by_user[u]] for y,u in ds]
			X, Y = vectorize(ds_docs, wrd2idx)
			basename = os.path.splitext(os.path.basename(name))[0]
			path = opts.out_folder + basename
			print "[saving data @ {}]".format(path)
			with open(path, "wb") as fid:
				cPickle.dump([X, Y, wrd2idx], fid, -1)
	
def get_parser():
	par = argparse.ArgumentParser(description="Extract Indices")
	par.add_argument('-labels_path', type=str, required=True, nargs='+', help='labeled data in the format: label TAB user_id')
	par.add_argument('-text_path', type=str, nargs='+',  help='text data in the format: user_id TAB text')
	par.add_argument('-out_folder', type=str, required=True, help='output folder')	
	par.add_argument('-cv', type=int, help='crossfold')
	par.add_argument('-users_only', action='store_true', help='only extract user info')
	par.add_argument('-cv_from', type=str, nargs='*', help="files for crossvalidation")
	par.add_argument('-embeddings', type=str, nargs='+', help='path to embeddings')	
	par.add_argument('-vocab_size', type=int, \
						help='max number of types to keep in the vocabulary')    		
	par.add_argument('-vocab_from', type=str, nargs='*', \
						help="compute vocabulary from these files")
	
	return par

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()
	
	print "[labels@: {} | text@: {} | out@: {} | user only: {}]".format(args.labels_path, 
														args.text_path, 
														args.out_folder,
														args.users_only)
	#create output folder if needed 
	args.out_folder = args.out_folder.rstrip("/") + "/"
	if not os.path.exists(os.path.dirname(args.out_folder)):
	    os.makedirs(os.path.dirname(args.out_folder))

	#loop through cross-validation folds (if any)
	if args.cv is None:
		# args.labels_path
		print "[computing vocabulary]"
		if args.vocab_from is not None:
			word_vocab, user_vocab = get_vocabularies(args.text_path, args.vocab_from, 
													 args.vocab_size, args.users_only)
		else:
			word_vocab, user_vocab = get_vocabularies(args.text_path, args.labels_path, 
													  args.vocab_size, args.users_only)

		main(args.text_path, args.labels_path, word_vocab, user_vocab, args)
	else:
		assert args.cv > 2, "need at leat 2 folds for cross-validation"
		# for cv_fold in xrange(1, args.cv+1):
		# 	if args.cv_from is None:
		# 		cv_fnames = [f+"_"+str(cv_fold) for f in args.input]
		# 	else:
		# 		cv_fnames = [f + "_" + str(cv_fold) for f in args.cv_from]
		# 	print "[computing vocabulary]"
		# 	if args.vocab_from is not None:
		# 		cv_vocab_fnames = [f+"_"+str(cv_fold) for f in args.vocab_from]				
		# 		vocabulary = get_vocabulary(cv_vocab_fnames, args.vocab_size)
		# 	else:
		# 		vocabulary = get_vocabulary([args.cv_fnames[0]], args.vocab_size)

		# 	main(cv_fnames, vocabulary, args)						
			
	#extract embeddings
	if args.embeddings is not None:
		for vecs_in in args.embeddings:
			print "[reading embeddings @ {}]".format(vecs_in)
			vecs_out = args.out_folder + os.path.basename(vecs_in)
			print "[saving embeddings @ {}]".format(vecs_out)
			embeddings.filter_embeddings(vecs_in, vecs_out, word_vocab)
