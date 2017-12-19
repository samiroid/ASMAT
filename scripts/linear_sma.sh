set -e
# Reset
COLOR_OFF='\033[0m'       # Text Reset
# Regular Colors
RED='\033[0;31m'          # RED
if [ -z "$1" ]
  then
    echo "please provide dataset"
    exit 1
fi
DATASET=$1

if [ -z "$2" ]
  then
	RUN_ID=$DATASET
else
	RUN_ID=$DATASET"_"$2
fi

echo "LINEAR SMA > " $DATASET
INPUT="/Users/samir/Dev/projects/phd_research/low_resource_sma/experiments/datasets/"$DATASET".txt"
BASE="/Users/samir/Dev/projects/phd_research/low_resource_sma/experiments/linear_sma/"$DATASET"/"
#create folders
mkdir -p $BASE
DATA=$BASE"/DATA/"
FEATURES=$BASE"/features/"
MODELS=$BASE"/models/"
RESULTS=$BASE"/results/"

#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"

CLEAN=1
if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $FEATURES/*.*
	rm -rf $MODELS/*.*
	rm -rf $DATA/*.*
fi

CLEAN_RESULTS=1
if (($CLEAN_RESULTS > 0)); then
	rm $RESULTS/*.*
fi

### DATA SPLIT ###
SPLIT=1
if (($SPLIT > 0)); then
	echo $RED"##### SPLIT DATA #####"$COLOR_OFF
	#first split the data into 80-20 split (temp/test)
	#then split the temp data into 80-20 split (train/dev)
	python ASMAT/toolkit/dataset_splitter.py -input $INPUT \
											 -output $DATA$DATASET"_tmp" $DATA$DATASET"_test" \
											 -rand_seed $RUN_ID &&
	python ASMAT/toolkit/dataset_splitter.py -input $DATA$DATASET"_tmp" \
											 -output $DATA$DATASET"_train" $DATA$DATASET"_dev" \
											 -rand_seed $RUN_ID
	rm -rf $DATA$DATASET"_tmp"
fi

### INDEX EXTRACTION ###
EXTRACT=1
if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX #####"$COLOR_OFF
	#extract vocabulary and indices
	python ASMAT/toolkit/extract.py -input $DATA$TRAIN $DATA$DEV -out_folder $FEATURES \
						            -save_vocab $FEATURES"vocabulary" 
	# #extract indices for test data (using the same vocabulary)
	python ASMAT/toolkit/extract.py -input $DATA$TEST -out_folder $FEATURES \
						    		-vocab_path $FEATURES"vocabulary" 
fi

### COMPUTE FEATURES ###
GET_FEATURES=1
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOW
	python ASMAT/toolkit/features.py -input $FEATURES$TRAIN $FEATURES$DEV $FEATURES$TEST \
							-out_folder $FEATURES -bow bin freq 	
fi

### LINEAR MODELS ###
LINEAR=1
if (($LINEAR > 0)); then
	VERBOSE=1
	echo $RED"##### LINEAR MODELS ##### "$COLOR_OFF
	python ASMAT/models/linear_model.py -verbose $VERBOSE -train $FEATURES$TRAIN \
							 -features BOW_bin -test $FEATURES$TEST \
							 -res_path $RESULTS"BOW.txt" 
	python ASMAT/models/linear_model.py -verbose $VERBOSE -train $FEATURES$TRAIN \
							 -features BOW_freq -test $FEATURES$TEST \
							 -res_path $RESULTS"BOW.txt"	
	python ASMAT/models/linear_model.py -verbose $VERBOSE -train $FEATURES$TRAIN \
							 -features BOW_freq BOW_bin -test $FEATURES$TEST \
							 -res_path $RESULTS"BOW.txt"	
fi