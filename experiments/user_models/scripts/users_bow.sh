set -e
# Reset
COLOR_OFF='\033[0m'       # Text Reset
# Regular Colors
RED='\033[0;31m'          # RED
if [ -z "$1" ]
  then
    echo "please provide tweets dataset"
    exit 1
fi
TWEETS=$1

if [ -z "$2" ]
  then
    echo "please provide labels dataset"
    exit 1
fi
DATASET=$2

if [ -z "$3" ]
  then
	RESFILE="user_models.txt"
	echo "default results file: " $RESFILE
else
	RESFILE=$3
fi

if [ -z "$4" ]
  then
	RUN_ID="BOW"
	echo "default RUN ID"
else
	RUN_ID=$4
fi
#config
PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments/user_models/"
DATA=$PROJECT_PATH"/DATA"
RESULTS=$DATA"/results/"$RESFILE
FEATURES=$DATA"/pkl/features"
MODELS=$DATA"/models"
#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"
#TWEETS=$DATASET"_users_tweets"

LINEAR_HYPERPARAMS=$PROJECT_PATH"/confs/linear.cfg"

echo "BOW USER SMA > " $DATASET
#OPTIONS
CLEAN=0
EXTRACT=1
GET_FEATURES=1
LINEAR_MODELS=1
HYPERPARAM=0
if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $DATA"/pkl" || True
fi

if [[ $HYPERPARAM -eq 0 ]]; then
	LINEAR_HYPERPARAMS="none"  
fi

### INDEX EXTRACTION ###
if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX #####"$COLOR_OFF
	#extract vocabulary and indices	
	#NOTE embedding based models can represent all words in the embedding matrix so it is 
	# ok to include the test set in the vocabulary
	python $PROJECT_PATH/code/users_extract.py -labels_path $DATA"/txt/"$TRAIN \
												$DATA"/txt/"$DEV $DATA"/txt/"$TEST \
										   		-text_path $DATA"/txt/"$TWEETS \
									-vocab_from $DATA"/txt/"$TRAIN $DATA"/txt/"$DEV \
									-idx_labels \
									-vocab_size 50000 \
									-out_folder $FEATURES 

fi

### COMPUTE FEATURES ###
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOE	
	python ASMAT/toolkit/features.py -input $FEATURES"/"$TRAIN $FEATURES"/"$DEV $FEATURES"/"$TEST \
							-out_folder $FEATURES \
							-bow bin freq \
							-sparse_bow
fi

### LINEAR MODELS ###
if (($LINEAR_MODELS > 0)); then
	echo $RED"##### LINEAR MODELS ##### "$COLOR_OFF	
	
	python ASMAT/toolkit/linear_model.py -features BOW-BIN \
										-run_id $RUN_ID \
										-train $FEATURES"/"$TRAIN \
										-test $FEATURES"/"$TEST \
										-dev $FEATURES"/"$DEV \
							 			-res_path $RESULTS \
										-hyperparams_path $LINEAR_HYPERPARAMS
							 
	python ASMAT/toolkit/linear_model.py -features BOW-FREQ	 \
										-run_id $RUN_ID \
										-train $FEATURES"/"$TRAIN \
										-test $FEATURES"/"$TEST \
										-dev $FEATURES"/"$DEV \
										-res_path $RESULTS \
										-hyperparams_path $LINEAR_HYPERPARAMS
	
fi

