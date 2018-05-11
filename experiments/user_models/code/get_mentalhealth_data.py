import codecs
from collections import defaultdict
import csv
import gzip
from ipdb import set_trace
import json
import os
import sys
from ASMAT.lib.data import preprocess, shuffle_split

#input files
path_train  = 'RAW_DATA/raw_datasets/mental_health/training_data/'
train_labels_path = "RAW_DATA/raw_datasets/mental_health/anonymized_user_info_by_chunk_training.csv"
output= 'experiments/user_models/DATA/txt/'
if not os.path.exists(output):
    os.makedirs(output)        
#### --- TRAIN DATA ---- ####
tweets_by_user = {}
print "[reading user tweets]"
z=0
MAX_USERS=100
# MAX_USERS=float('inf')
MIN_TWEETS=100
for fname in os.listdir(path_train):	
	if os.path.splitext(path_train+fname)[1]!=".gz":
			print "ignored %s"% fname 
			continue			
	with gzip.open(path_train+fname, 'r') as f:			
		user = fname[:fname.index(".")] 		
		data = [preprocess(json.loads(l)['text']) for l in f]
		# data = set([json.loads(l)['text'] for l in f])
		if len(data) < MIN_TWEETS:
			print "ignored user %s | %d tweets" % (user, len(data))
			continue		
		tweets_by_user[user] = set(data)
		z+=1	
	sys.stdout.write("\ruser: "+user+" ("+ str(z) +")"+" "*20)
	sys.stdout.flush()
	if z>=MAX_USERS:
		print "out early!!!!"
		break	

print "[writing user tweets]"
user_corpus = codecs.open(output+"mental_health_tweets","w","utf-8")
for user, twt in tweets_by_user.items():
	# set_trace()
	tweets = ' '.join(twt)
	tweets = ' '.join(tweets.split())
	user_corpus.write(u"{}\t{}\n".format(user, tweets))

print "[writing tweets for word embedding training]"
user_corpus = codecs.open(output+"word_embeddings_corpus","w","utf-8")
for user, twt in tweets_by_user.items():
	# set_trace()
	tweets = '\n'.join(twt)	
	user_corpus.write(u"{}\n".format(tweets))

print "[reading training labels]"
ptsd = {}
depression = {}
with open(train_labels_path) as fid:
	f = csv.reader(fid)
	f.next() #skip the header
	for r in f:
		user = r[0]
		cond = r[4]
		if cond == "ptsd":
			ptsd[user] = cond
		elif cond == "depression":
			depression[user] = cond
		elif cond == "control":
			ptsd[user] = cond
			depression[user] = cond
		

#stratified split
ptsd_tuples = [[x[1],x[0]] for x in ptsd.items()]
tmp_set, ptsd_test = shuffle_split(ptsd_tuples)
ptsd_train, ptsd_dev = shuffle_split(tmp_set)

depression_tuples = [[x[1],x[0]] for x in depression.items()]
tmp_set, depression_test = shuffle_split(depression_tuples)
depression_train, depression_dev = shuffle_split(tmp_set)

print "[writing PTSD data]"
with open(output+"ptsd_train","w") as fod:
	for label, user in ptsd_train:		
		if user not in tweets_by_user: 
			# print "unknown dude %s" % user		
			continue
		fod.write("{}\t{}\n".format(label, user))
		
with open(output+"ptsd_test","w") as fod:
	for label, user in ptsd_test:
		if user not in tweets_by_user: 
			# print "unknown dude %s" % user		
			continue		
		fod.write("{}\t{}\n".format(label, user))
		
with open(output+"ptsd_dev","w") as fod:
	for label, user in ptsd_dev:		
		if user not in tweets_by_user: 
			# print "unknown dude %s" % user		
			continue		
		fod.write("{}\t{}\n".format(label, user))
		
print "[writing DEPRESSION data]"
with open(output+"depression_train","w") as fod:
	for label, user in depression_train:		
		if user not in tweets_by_user: 
			# print "unknown dude %s" % user		
			continue
		fod.write("{}\t{}\n".format(label, user))
		
with open(output+"depression_test","w") as fod:
	for label, user in depression_test:
		if user not in tweets_by_user: 
			# print "unknown dude %s" % user		
			continue		
		fod.write("{}\t{}\n".format(label, user))
		
with open(output+"depression_dev","w") as fod:
	for label, user in depression_dev:		
		if user not in tweets_by_user: 
			# print "unknown dude %s" % user		
			continue		
		fod.write("{}\t{}\n".format(label, user))
		