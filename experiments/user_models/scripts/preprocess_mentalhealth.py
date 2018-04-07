import codecs
import csv
import gzip
from ipdb import set_trace
import json
import os
import sys
from ASMAT.lib.data import preprocess
# sys.path.append("code")
# from sma_toolkit.preprocess import preprocess
#input files
path_train  = 'DATA/raw_datasets/mental_health/training_data/'
train_labels_path = "DATA/raw_datasets/mental_health/anonymized_user_info_by_chunk_training.csv"
output= 'DATA/raw_datasets/mental_health/txt/'
if not os.path.exists(output):
    os.makedirs(output)        
#### --- TRAIN DATA ---- ####
train_tweets = {}
print "reading user tweets..."
z=0
MAX_USERS=10
MIN_TWEETS=100
for fname in os.listdir(path_train):	
	if os.path.splitext(path_train+fname)[1]!=".gz":
			print "ignored %s"% fname 
			continue			
	with gzip.open(path_train+fname, 'r') as f:			
		user = fname[:fname.index(".")] 		
		data = set([preprocess(json.loads(l)['text']) for l in f])
		# data = set([json.loads(l)['text'] for l in f])
		if len(data) < MIN_TWEETS:
			print "ignored user %s | %d tweets" % (user, len(data))
			continue
		train_tweets[user] = set(data)
		z+=1	
	sys.stdout.write("\ruser: "+user+" ("+ str(z) +")"+" "*20)
	sys.stdout.flush()
	if z>=MAX_USERS:
		print "out early!!!!"
		break		
train_labels = {}
print "reading training labels..."
with open(train_labels_path) as fid:
	f = csv.reader(fid)
	f.next() #skip the header
	for r in f:
		user = r[0]
		cond = r[4]		
		train_labels[user] = cond		

train_data  = codecs.open(output+"train_data.txt","w","utf-8")
#file with all the text (to learn the embeddings)
corpus = codecs.open(output+"corpus.txt","w","utf-8")
#here all the tweets of the same user are grouped into one document (each line corresponds to a user)
user_corpus = codecs.open(output+"user_bows.txt","w","utf-8")

print "writing training data..."
for j, (user, condition) in enumerate(train_labels.items()):			
	if user not in train_tweets: 
		print "unknown dude %s" % user		
		continue
	if condition not in {'ptsd':0,'depression':0,'control':0}:
		print "unknown condition:%s (user:%s)" % (condition,user)		
		continue
	#write in user corpus
	for t in train_tweets[user]: corpus.write(u"%s\t%s\n" % (user,t))		
	tweets = ' '.join(train_tweets[user])
	assert type(tweets) == unicode
	user_corpus.write(u"%s\t%s\n" % (user,tweets))
	train_data.write(u"%d\t%s\t%d\t%s\t%s\n" % (j,user,len(train_tweets[user]),condition,tweets))
	# if  condition == 'depression':				 
	# 	 train_depression.write(u"%d\t%s\t%d\t%s\t%s\n" % (j,user,len(train_tweets[user]),condition,tweets))
	# 	 train_ptsd_vs_dep.write(u"%d\t%s\t%d\t%s\t%s\n" % (j,user,len(train_tweets[user]),condition,tweets))
	# elif condition == 'ptsd':				 
	# 	 train_ptsd.write(u"%d\t%s\t%d\t%s\t%s\n" % (j,user,len(train_tweets[user]),condition, tweets))
	# 	 train_ptsd_vs_dep.write(u"%d\t%s\t%d\t%s\t%s\n" % (j,user,len(train_tweets[user]),condition,tweets))
	# elif condition == 'control': 				 
	# 	 train_depression.write(u"%d\t%s\t%d\t%s\t%s\n" % (j,user,len(train_tweets[user]),condition,tweets))
	# 	 train_ptsd.write(u"%d\t%s\t%d\t%s\t%s\n" % (j,user,len(train_tweets[user]),condition, tweets))
	
		
#close all the files
# train_depression.close()
# train_ptsd.close()
corpus.close()
user_corpus.close()
