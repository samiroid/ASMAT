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

PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments"
BASE=$PROJECT_PATH"/linear_sma/"$DATASET

#create folders
mkdir -p $BASE
DATA=$BASE"/DATA"
FEATURES=$BASE"/features"
MODELS=$BASE"/models"
RESULTS=$BASE"/results"
#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"

echo "LINEAR SMA > " $DATASET
#OPTIONS
CLEAN=0
SPLIT=0
EXTRACT=1
GET_FEATURES=0
LINEAR=0

if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $FEATURES/*.* || True
	rm -rf $MODELS/*.* || True
	rm -rf $DATA/*.* || True
	rm $RESULTS/*.* || True
fi
### DATA SPLIT ###
if (($SPLIT > 0)); then
	echo $RED"##### SPLIT DATA #####"$COLOR_OFF
	#first split the data into 80-20 split (temp/test)
	#then split the temp data into 80-20 split (train/dev)
	INPUT=$PROJECT_PATH"/datasets/"$DATASET".txt"
	python ASMAT/toolkit/dataset_splitter.py -input $INPUT \
											 -output $DATA"/"$DATASET"_tmp" $DATA"/"$DATASET"_test" \
											 -rand_seed $RUN_ID 
	python ASMAT/toolkit/dataset_splitter.py -input $DATA"/"$DATASET"_tmp" \
											 -output $DATA"/"$DATASET"_train" $DATA"/"$DATASET"_dev" \
											 -rand_seed $RUN_ID
	rm -rf $DATA"/"$DATASET"_tmp"
fi
### INDEX EXTRACTION ###
if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX #####"$COLOR_OFF
	#extract vocabulary and indices
	python ASMAT/toolkit/extract.py -input $DATA"/"$TRAIN $DATA"/"$DEV $DATA"/"$TEST \
									-out_folder $FEATURES \
									-vocab_from $DATA"/"$TRAIN $DATA"/"$DEV \
									-idx_labels	
fi
### COMPUTE FEATURES ###
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOW
	python ASMAT/toolkit/features.py -input $FEATURES"/"$TRAIN $FEATURES"/"$DEV $FEATURES"/"$TEST \
							-out_folder $FEATURES -bow bin freq 	
fi

### LINEAR MODELS ###
if (($LINEAR > 0)); then
	echo $RED"##### LINEAR MODELS ##### "$COLOR_OFF
	python ASMAT/models/linear_model.py -train $FEATURES"/"$TRAIN \
							 -features BOW_bin -test $FEATURES"/"$TEST \
							 -res_path $RESULTS"/BOW.txt" 
	python ASMAT/models/linear_model.py -train $FEATURES"/"$TRAIN \
							 -features BOW_freq -test $FEATURES"/"$TEST \
							 -res_path $RESULTS"/BOW.txt"	
	python ASMAT/models/linear_model.py -train $FEATURES"/"$TRAIN \
							 -features BOW_freq BOW_bin -test $FEATURES"/"$TEST \
							 -res_path $RESULTS"/BOW.txt"	
fi