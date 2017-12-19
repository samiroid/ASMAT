import argparse
import codecs
import cPickle 
from ipdb import set_trace
import os 
import sys
sys.path.append("..")

from ASMAT.lib.extract import docs2idx, build_vocabulary
from ASMAT.lib import embeddings
from ASMAT.lib.data import read_dataset, flatten_list, filter_labels

def get_parser():
    parser = argparse.ArgumentParser(description="Extract Indices")
    parser.add_argument('-input', type=str, required=True, nargs='+',
                        help='train data')  
    parser.add_argument('-out_folder',type=str, required=True,
                        help='output folder')						
    parser.add_argument('-vectors', type=str, nargs='+',
                        help='path to embeddings')  
    parser.add_argument('-labels', type=str, nargs='+',
                        help='label set')  
    parser.add_argument('-vocab_size', type=int, 
                        help='max number of types to keep in the vocabulary')
    parser.add_argument('-vocab_path', type=str, 
                        help='path to a precomputed vocabulary')
    parser.add_argument('-save_vocab', type=str, 
                        help='path to save vocabulary')    
    
    return parser

if __name__=="__main__":	
	parser = get_parser()
	args = parser.parse_args()		
	datasets = []
	if args.labels is not None:
		print "[labels: {}]".format(repr(args.labels))
	for dataset in args.input:
		print("[reading data @ {}]".format(repr(dataset)))	
		ds = read_dataset(dataset, labels=args.labels)		
		datasets.append(ds)
		# doc_sets.append(docs)
		# label_sets.append(labels)
	#flatten the list of lists 
	flat_dataset = flatten_list(datasets)
	docs = [x[1] for x in flat_dataset]
	labels = [x[0] for x in flat_dataset]	
	# docs   = [d for sublist in doc_sets for d in sublist]
	# labels = [l for sublist in label_sets for l in sublist]	
	#label map and vocabulary
	label_map = build_vocabulary(labels)  		
	if args.vocab_path is not None:
		print "[opening vocabulary @ {}]".format(args.vocab_path)
		with open(args.vocab_path) as fid:
			vocabulary = cPickle.load(fid)
	else: 
		vocabulary = build_vocabulary(docs,max_words=args.vocab_size)	
	#create output folder if needed
	if not os.path.exists(os.path.dirname(args.out_folder)):
	    os.makedirs(os.path.dirname(args.out_folder))   	

	if args.save_vocab is not None:		
		#get the basename (will use the basename of the first dataset)
		# basename = os.path.splitext(os.path.basename(args.input[0]))[0]
		# path = args.out_folder+basename+"_vocabulary.pkl"
		print "[saving vocabulary @ {}]".format(args.save_vocab)
		with open(args.save_vocab,"wb") as fid:
			cPickle.dump(vocabulary, fid, -1)
	
	if args.vectors is not None:
		for vecs_in in args.vectors:
			print "[reading embeddings @ {}]".format(vecs_in)
			vecs_basename = os.path.splitext(os.path.basename(vecs_in))[0]
			#get the basename (will use the basename of the first dataset)
			dataset_basename = os.path.splitext(os.path.basename(args.input[0]))[0]
			vecs_out = args.out_folder+"vectors_"+vecs_basename+".txt"
			print "[saving embeddings @ {}]".format(vecs_out)
			embeddings.filter_embeddings(vecs_in, vecs_out, vocabulary)	   

	print "[converting docs to indices]"	
	for name, ds in zip(args.input, datasets): 
		docs   = [x[1] for x in ds]
		labels = [x[0] for x in ds]
		Y = [label_map[l] for l in labels]
		X, _ = docs2idx(docs, vocabulary)			
		basename = os.path.splitext(os.path.basename(name))[0]
		path = args.out_folder+basename
		print "[saving data @ {}]".format(path)
		with open(path,"wb") as fid:
			cPickle.dump([X, Y, vocabulary, label_map], fid, -1)
