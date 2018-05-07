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
#output= 'DATA/raw_datasets/mental_health/txt/'
output= 'experiments/user_models/DATA/txt/'
if not os.path.exists(output):
    os.makedirs(output)        
#### --- TRAIN DATA ---- ####
tweets_by_user = {}
print "[reading user tweets]"
z=0
MAX_USERS=100
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
user_corpus = codecs.open(output+"mental_health_users_tweets","w","utf-8")
print "[writing user tweets]"
for user, twt in tweets_by_user.items():
	# set_trace()
	tweets = ' '.join(twt)
	tweets = ' '.join(tweets.split())
	user_corpus.write(u"{}\t{}\n".format(user, tweets))

print "[reading training labels]"
train_labels = {}
with open(train_labels_path) as fid:
	f = csv.reader(fid)
	f.next() #skip the header
	for r in f:
		user = r[0]
		cond = r[4]		
		train_labels[user] = cond		

#stratified split
train_tuples = [[x[1],x[0]] for x in train_labels.items()]
tmp_set, test_set = shuffle_split(train_tuples)
train_set, dev_set = shuffle_split(tmp_set)
print "[writing training data]"

with open(output+"mental_health_train","w") as fod:
	for label, user in train_set:		
		if user not in tweets_by_user: 
			# print "unknown dude %s" % user		
			continue
		if label not in ('ptsd','depression','control'):
			# print "unknown condition:%s (user:%s)" % (label,user)
			continue
		print "train > x: {} | y: {}".format(user, label)
		fod.write("{}\t{}\n".format(label, user))
		
with open(output+"mental_health_test","w") as fod:
	for label, user in test_set:
		if user not in tweets_by_user: 
			# print "unknown dude %s" % user		
			continue
		if label not in ('ptsd','depression','control'):
			# print "unknown condition:%s (user:%s)" % (label,user)
			continue
		print "test > x: {} | y: {}".format(user, label)
		fod.write("{}\t{}\n".format(label, user))
		
with open(output+"mental_health_dev","w") as fod:
	for label, user in dev_set:		
		if user not in tweets_by_user: 
			# print "unknown dude %s" % user		
			continue
		if label not in ('ptsd','depression','control'):
			# print "unknown condition:%s (user:%s)" % (label,user)
			continue		
		print "dev > x: {} | y: {}".format(user, label)
		fod.write("{}\t{}\n".format(label, user))
		