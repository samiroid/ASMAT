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
	RESFILE="asma.txt"
	echo "default results file"
else
	RESFILE=$2
fi

#config
PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments/asma"
DATA=$PROJECT_PATH"/DATA"
RESULTS=$DATA"/results/"$RESFILE
LINEAR_FEATURES=$DATA"/pkl/linear_features"
MODELS=$DATA"/models"
#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"

LINEAR_HYPERPARAMS=$PROJECT_PATH"/confs/linear.cfg"

echo "BOW SMA > " $DATASET
#OPTIONS
CLEAN=1
EXTRACT=1
GET_FEATURES=1
LINEAR_MODELS=1
HYPERPARAM=0

if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $DATA"/pkl" || True
fi

if [[ $HYPERPARAM -eq 0 ]]; then	
	LINEAR_HYPERPARAMS="a"  
fi


### INDEX EXTRACTION ###
if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX #####"$COLOR_OFF
	#extract vocabulary and indices
	#NOTE: linear models need to build vocabulary only from train+dev data
	python ASMAT/toolkit/extract.py -input $DATA"/txt/"$TRAIN $DATA"/txt/"$DEV \
										   $DATA"/txt/"$TEST \
									-vocab_from $DATA"/txt/"$TRAIN $DATA"/txt/"$DEV \
									-out_folder $LINEAR_FEATURES \
									-idx_labels	
fi
### COMPUTE FEATURES ###
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOW
	python ASMAT/toolkit/features.py -input $LINEAR_FEATURES"/"$TRAIN $LINEAR_FEATURES"/"$DEV $LINEAR_FEATURES"/"$TEST \
									-bow bin \
									-out_folder $LINEAR_FEATURES 
fi

### LINEAR MODELS ###
if (($LINEAR_MODELS > 0)); then
	echo $RED"##### LINEAR MODELS ##### "$COLOR_OFF	
	
	python ASMAT/toolkit/linear_model.py -features BOW-BIN \
										-train $LINEAR_FEATURES"/"$TRAIN \
							  			-test $LINEAR_FEATURES"/"$TEST \
										-dev $LINEAR_FEATURES"/"$DEV \
							 			-res_path $RESULTS \
										-hyperparams_path $LINEAR_HYPERPARAMS 
										

	python ASMAT/toolkit/linear_model.py -features naive_bayes \
										-train $LINEAR_FEATURES"/"$TRAIN \
							  			-test $LINEAR_FEATURES"/"$TEST \
										-dev $LINEAR_FEATURES"/"$DEV \
							 			-res_path $RESULTS 										 
fi