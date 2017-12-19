import argparse
from bs4 import BeautifulSoup
from ipdb import set_trace
import os
import sys
sys.path.append("..")

from ASMAT.lib import data as data_reader
from ASMAT.lib.helpers import str2seed

def get_parser():
	par = argparse.ArgumentParser(description="Split Dataset")
	par.add_argument('-input', type=str, required=True, help='input data')
	par.add_argument('-output', type=str, required=True, nargs=2, help='output files')
	par.add_argument('-split', type=float, default=0.8, help='data split')
	par.add_argument('-no_strat', action="store_true", help="do not stratified data for split")
	par.add_argument('-rand_seed', type=str, default="1234", help='randomization seed')

	return par

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()	
	try:
		seed = int(args.rand_seed)
	except ValueError:		
		seed = str2seed(args.rand_seed)	
	print "[seed ({}) | input: {} | split: {} | strat: {}]".format(seed, args.input, \
																	args.split, \
																	not args.no_strat)	
	#fname, ext = os.path.splitext(args.input)	
	fname_1 = args.output[0]
	fname_2 = args.output[1]
	data = data_reader.read_dataset(args.input)	
	if args.no_strat:
		data_1, data_2 = data_reader.simple_split(data, args.split, random_seed=seed)
	else:
		data_1, data_2 = data_reader.shuffle_split(data, args.split, random_seed=seed)
	
	print "[saving: {} | {} ]".format(fname_1, fname_2)	
	data_reader.save_dataset(data_1, fname_1)
	data_reader.save_dataset(data_2,fname_2)
