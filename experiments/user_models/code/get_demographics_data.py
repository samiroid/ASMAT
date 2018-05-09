import codecs
from collections import defaultdict
import csv
import gzip
from ipdb import set_trace
import json
import os
import sys
from ASMAT.lib.data import preprocess, shuffle_split

data_path = "RAW_DATA/raw_datasets/demographics/user_demographics_labels.txt"
output= 'experiments/user_models/DATA/txt/'
age_dataset = []
gender_dataset = []
race_dataset = []
MAX_USERS=100
MAX_USERS=float('inf')
print "[reading labels]"
with open(data_path) as fid:
	fid.next()
	i=0
	for l in fid:
		i+=1
		if i>=MAX_USERS: 
			print "break early"
			break
		user, gender, age, race = l.replace("\n","").split("\t")
		age_dataset.append([age, user])
		gender_dataset.append([gender, user])
		race_dataset.append([race, user])

print "[saving age dataset]"
tmp_set, test_set = shuffle_split(age_dataset)
train_set, dev_set = shuffle_split(tmp_set)
with open(output+"demos_age_train","w") as fod:
	for label, user in train_set:						
		fod.write("{}\t{}\n".format(label, user))
with open(output+"demos_age_test","w") as fod:
	for label, user in test_set:						
		fod.write("{}\t{}\n".format(label, user))
with open(output+"demos_age_dev","w") as fod:
	for label, user in dev_set:						
		fod.write("{}\t{}\n".format(label, user))


print "[saving gender dataset]"
tmp_set, test_set = shuffle_split(gender_dataset)
train_set, dev_set = shuffle_split(tmp_set)
with open(output+"demos_gender_train","w") as fod:
	for label, user in train_set:						
		fod.write("{}\t{}\n".format(label, user))
with open(output+"demos_gender_test","w") as fod:
	for label, user in test_set:						
		fod.write("{}\t{}\n".format(label, user))
with open(output+"demos_gender_dev","w") as fod:
	for label, user in dev_set:						
		fod.write("{}\t{}\n".format(label, user))

print "[saving race dataset]"
tmp_set, test_set = shuffle_split(race_dataset)
train_set, dev_set = shuffle_split(tmp_set)
with open(output+"demos_race_train","w") as fod:
	for label, user in train_set:						
		fod.write("{}\t{}\n".format(label, user))
with open(output+"demos_race_test","w") as fod:
	for label, user in test_set:						
		fod.write("{}\t{}\n".format(label, user))
with open(output+"demos_race_dev","w") as fod:
	for label, user in dev_set:						
		fod.write("{}\t{}\n".format(label, user))

