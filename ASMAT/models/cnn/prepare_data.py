#prepare training data
import my_utils as utils
from my_utils.embeddings import save_embeddings_txt
EMBEDDING_IN = "DATA/embeddings/str_skip_600.txt"
FILTERED_EMB = "DATA/embeddings/filtered_embedding.txt"

def write_data(file_in, file_out):
	with open(file_in,"r") as fid:
		with open(file_out,"w") as fod:
			for m in fid:
				label = m.split("\t")[2]
				doc = m.split("\t")[4]
				if label=="positive":
					l=1
				elif label=="negative":
					l=-1
				elif label=="neutral":
					l=0
				else:					
					raise NotImplementedError, "unknown label " + label

				fod.write("%d\t%s" % (l,doc))

def write_all_data():
	"""
		Write all files in a simplified format
	"""
	file_in = "DATA/input/semeval_train+aliak_postap_noextra.txt"
	file_out = "DATA/txt/semeval_train.txt"
	write_data(file_in, file_out)

	file_in = "DATA/input/tweets_2013.txt"
	file_out = "DATA/txt/tweets_2013.txt"
	write_data(file_in, file_out)

	file_in = "DATA/input/tweets_2014.txt"
	file_out = "DATA/txt/tweets_2014.txt"
	write_data(file_in, file_out)

	file_in = "DATA/input/tweets_2015.txt"
	file_out = "DATA/txt/tweets_2015.txt"
	write_data(file_in, file_out)

def prepare_embeddings():

	datasets = ["DATA/txt/tweets_2015.txt",
				"DATA/txt/tweets_2014.txt",
				"DATA/txt/tweets_2013.txt",
				"DATA/txt/train.txt"]

	all_msgs = []
	for d in datasets:
		with open(d,"r") as fid:
			msgs = [m.split("\t")[1] for m in fid]
		all_msgs+=msgs

	wrd2idx = utils.word_2_idx(all_msgs)
	save_embeddings_txt(EMBEDDING_IN, FILTERED_EMB, wrd2idx)

	

prepare_embeddings()