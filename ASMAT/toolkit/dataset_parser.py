import argparse
from bs4 import BeautifulSoup
from ipdb import set_trace
import os
import sys
sys.path.append("..")
from ASMAT.lib.preprocess import preprocess
from ASMAT.lib.data import save_dataset, flatten_list
import codecs 

SEP_EMOJI=False

def omd_hcr(in_path):
	data = []
	with open(in_path) as fid:
		soup = BeautifulSoup(fid.read(), "xml")
		for item in soup.findAll('item'):
			if item.attrs['label'] in ['positive', 'negative', 'neutral']:
				msg = item.find("content").text
				msg = preprocess(msg.decode("utf-8"),sep_emoji=SEP_EMOJI)
				data.append([item.attrs['label'], msg])	
	return data

def stance(in_path, topic):
	data = []
	with open(in_path) as fid:
		for l in fid:
			spt = l.replace("\r\n", "").split("\t")
			current_topic, tweet, label = spt[1:]
			if current_topic == topic:
				if label in ['FAVOR', 'AGAINST', 'NONE']:
					tweet = preprocess(tweet.decode("utf-8"))
					ex = [label.lower(), tweet.encode("utf-8")]
					data.append(ex)
	return data 

def semeval(in_path):
	data = []
	with codecs.open(in_path, "r","utf-8") as fid:
		for l in fid:
			spt = l.replace("\n", "").split("\t")
			label = spt[0].replace("\"", "")
			tweet = spt[1] #.decode("utf-8")
			if label in ['positive', 'negative', 'neutral']:
				tweet = preprocess(tweet,sep_emoji=SEP_EMOJI) #.encode("utf-8")
				ex = (label, tweet)
				data.append(ex)
	return data

def casm(in_path):
	cache = dict()
	data = []
	labs = []
	with codecs.open(in_path, "r","utf-8") as fid:
		for l in fid:
			spt = l.split("\t")
			label = spt[0].split(",")[1]
			tweet = preprocess(spt[1],sep_emoji=SEP_EMOJI)
			if tweet in cache:
				continue
			cache[tweet] = True
			# print label, tweet
			ex = [label, tweet]
			labs.append(label)
			data.append(ex)
	print "labels: ", list(set(labs))
	return data

def get_parser():
	par = argparse.ArgumentParser(description="Parse common SMA datasets")
	par.add_argument('-input', type=str, required=True, nargs='+', help='input data')
	par.add_argument('-outfolder', type=str, required=True, help='output folder')
	par.add_argument('-format', type=str, required=True, choices=["omd_hcr", "semeval", "casm", "stance"], help='data format')
	par.add_argument('-labels', type=str, nargs='+', help='label set')
	par.add_argument('-outname', type=str, help='if set, merge all the input files into a single output file')	
	return par

if __name__ == "__main__":
	cmdline_parser = get_parser()
	args = cmdline_parser.parse_args()
	data_parser = None
	if args.format == "omd_hcr":
		data_parser = omd_hcr
	elif args.format == "semeval":
		data_parser = semeval
	elif args.format == "casm":
		data_parser = casm
	else:
		raise NotImplementedError
	data = []
	for ds in args.input:
		print "[reading: {}]".format(ds)
		X = data_parser(ds)		
		data.append(X)
	
	if args.outname is not None:
		flat_data = flatten_list(data)
		save_dataset(flat_data, args.outfolder + "/" + args.outname, args.labels)
	else:
		for ds, x in zip(args.input, data):
			basename = os.path.basename(ds)
			fname = args.outfolder + "/" + basename
			save_dataset(x, fname, args.labels)
