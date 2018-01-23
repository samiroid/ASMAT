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
#config
PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments"
BASE=$PROJECT_PATH"/sma/CV_linear/"$DATASET
HYPERPARAMS=$PROJECT_PATH"/confs/default.cfg"
RESULTS=$PROJECT_PATH"/sma/CV_linear/results/"$DATASET".txt"
RESULTS=$PROJECT_PATH"/sma/CV_linear/results/lowres_sma.txt"
#create folders
mkdir -p $BASE
DATA=$BASE"/DATA"
FEATURES=$BASE"/features"
MODELS=$BASE"/models"
#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"

echo "LINEAR SMA > " $DATASET
#OPTIONS
CLEAN=1
SPLIT=1
EXTRACT=1
GET_FEATURES=1
LINEAR=1
CV=10

if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $BASE || True
fi


if (($SPLIT > 0)); then
	echo $RED"##### SPLIT DATA #####"$COLOR_OFF
	
	INPUT=$PROJECT_PATH"/datasets/"$DATASET".txt"
	python ASMAT/toolkit/dataset_splitter.py -input $INPUT \
											 -train $DATA"/"$DATASET"_train" \
											 -test $DATA"/"$DATASET"_test" \
											 -dev $DATA"/"$DATASET"_dev" \
											 -cv $CV \
											 -rand_seed $RUN_ID 
	
fi

### INDEX EXTRACTION ###
if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX #####"$COLOR_OFF
	#extract vocabulary and indices
	python ASMAT/toolkit/extract.py -input $DATA"/"$TRAIN $DATA"/"$DEV $DATA"/"$TEST \
									-vocab_from $DATA"/"$TRAIN $DATA"/"$DEV \
									-out_folder $FEATURES \
									-cv $CV \
									-idx_labels	
fi
### COMPUTE FEATURES ###
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOW
	python ASMAT/toolkit/features.py -input $FEATURES"/"$TRAIN $FEATURES"/"$DEV $FEATURES"/"$TEST \
									-bow bin \
									-out_folder $FEATURES \
									-cv $CV 
fi

### LINEAR MODELS ###
if (($LINEAR > 0)); then
	echo $RED"##### LINEAR MODELS ##### "$COLOR_OFF	
	
	python ASMAT/models/linear_model.py -features BOW-BIN \
										-train $FEATURES"/"$TRAIN \
							  			-test $FEATURES"/"$TEST \
										-dev $FEATURES"/"$DEV \
										-cv $CV \
							 			-res_path $RESULTS

	python ASMAT/models/linear_model.py -features naive_bayes \
										-train $FEATURES"/"$TRAIN \
							  			-test $FEATURES"/"$TEST \
										-dev $FEATURES"/"$DEV \
										-cv $CV \
							 			-res_path $RESULTS
fi