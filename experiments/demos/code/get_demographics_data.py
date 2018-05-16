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
output = 'experiments/demos/DATA/txt/'
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
		user, gender, age, race = l.replace("\n","").split("\t")
		#ignore "others"
		if race == "O": continue		
		race_dataset.append([race, user])
		#cluster childs and teens 
		if age == "child":   age = "teen" 
		#cluster elders and 50s
		elif age == "elder": age = "50s"		
		age_dataset.append([age, user])
		gender_dataset.append([gender, user])		
		i+=1
		if i>=MAX_USERS: 
			print "break early"
			break
print "[saving age dataset]"
tmp_set, test_set = shuffle_split(age_dataset)
train_set, dev_set = shuffle_split(tmp_set)
cohort_all = open(output+"cohort.txt","w")
age_all = open(output+"cohort_age.txt","w")
age_all.write("user\tage\n")
with open(output+"age_train","w") as fod:
	for label, user in train_set:						
		fod.write("{}\t{}\n".format(label, user))
		age_all.write("{}\t{}\n".format(user, label))
		cohort_all.write("{}\t{}\n".format(user, user))
with open(output+"age_test","w") as fod:
	for label, user in test_set:						
		fod.write("{}\t{}\n".format(label, user))
		age_all.write("{}\t{}\n".format(user, label))
		cohort_all.write("{}\t{}\n".format(user, user))
with open(output+"age_dev","w") as fod:
	for label, user in dev_set:						
		fod.write("{}\t{}\n".format(label, user))
		age_all.write("{}\t{}\n".format(user, label))
		cohort_all.write("{}\t{}\n".format(user, user))
age_all.close()
cohort_all.close()

print "[saving gender dataset]"
tmp_set, test_set = shuffle_split(gender_dataset)
train_set, dev_set = shuffle_split(tmp_set)
gender_all = open(output+"cohort_gender.txt","w")
gender_all.write("user\tgender\n")
with open(output+"gender_train","w") as fod:
	for label, user in train_set:						
		fod.write("{}\t{}\n".format(label, user))
		gender_all.write("{}\t{}\n".format(user, label))
with open(output+"gender_test","w") as fod:
	for label, user in test_set:						
		fod.write("{}\t{}\n".format(label, user))
		gender_all.write("{}\t{}\n".format(user, label))
with open(output+"gender_dev","w") as fod:
	for label, user in dev_set:						
		fod.write("{}\t{}\n".format(label, user))
		gender_all.write("{}\t{}\n".format(user, label))
gender_all.close()
print "[saving race dataset]"
tmp_set, test_set = shuffle_split(race_dataset)
train_set, dev_set = shuffle_split(tmp_set)
race_all = open(output+"cohort_race.txt","w")
race_all.write("user\trace\n")
with open(output+"race_train","w") as fod:
	for label, user in train_set:						
		fod.write("{}\t{}\n".format(label, user))
		race_all.write("{}\t{}\n".format(user, label))
with open(output+"race_test","w") as fod:
	for label, user in test_set:						
		fod.write("{}\t{}\n".format(label, user))
		race_all.write("{}\t{}\n".format(user, label))
with open(output+"race_dev","w") as fod:
	for label, user in dev_set:						
		fod.write("{}\t{}\n".format(label, user))
		race_all.write("{}\t{}\n".format(user, label))
race_all.close()
