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
BASE=$PROJECT_PATH"/sma/neural/"$DATASET
RESULTS=$PROJECT_PATH"/sma/results/"$DATASET".txt"
RESULTS=$PROJECT_PATH"/sma/results/lowres_sma.txt"
HYPERPARAMS=$PROJECT_PATH"/confs/default.cfg"

#create folders
mkdir -p $BASE
DATA=$BASE"/DATA"
FEATURES=$BASE"/features"
MODELS=$BASE"/models"

#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"
EMBEDDINGS="DATA/embeddings/str_skip_50.txt"
FILTERED_EMBEDDINGS=$FEATURES"/str_skip_50.txt"

#OPTIONS
CLEAN=0
SPLIT=1
EXTRACT=1
GET_FEATURES=1
LINEAR=1
NLSE=0

echo "NEURAL SMA > " $DATASET

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
											 -rand_seed $RUN_ID &&
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
									-vocab_from $DATA"/"$TRAIN $DATA"/"$DEV $DATA"/"$TEST \
									-out_folder $FEATURES \
									-idx_labels \
									-embeddings $EMBEDDINGS 
fi


### COMPUTE FEATURES ###
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOE
	python ASMAT/toolkit/features.py -input $FEATURES"/"$TRAIN $FEATURES"/"$DEV $FEATURES"/"$TEST \
							-out_folder $FEATURES \
							-boe bin sum \
							-embeddings $FILTERED_EMBEDDINGS	
	
	python ASMAT/toolkit/features.py -input $FEATURES"/"$TRAIN \
							-out_folder $FEATURES \
							-nlse \
							-embeddings $FILTERED_EMBEDDINGS	
fi

### LINEAR MODELS ###
if (($LINEAR > 0)); then
	echo $RED"##### LINEAR MODELS ##### "$COLOR_OFF
	python ASMAT/models/linear_model.py -train $FEATURES"/"$TRAIN \
							 -features BOE_bin -test $FEATURES"/"$TEST \
							 -res_path $RESULTS
	python ASMAT/models/linear_model.py -train $FEATURES"/"$TRAIN \
							 -features BOE_sum -test $FEATURES"/"$TEST \
							 -res_path $RESULTS
	python ASMAT/models/linear_model.py -train $FEATURES"/"$TRAIN \
							 -features BOE_bin BOE_sum -test $FEATURES"/"$TEST \
							 -res_path $RESULTS
fi

# ### NLSE #####
if (($NLSE > 0)); then
	echo $RED"##### NLSE ##### "$COLOR_OFF
	python ASMAT/models/train_nlse.py -tr $FEATURES"/"$TRAIN"_NLSE.pkl" \
							   		   -dev $FEATURES"/"$DEV \
                           	   		   -ts $FEATURES"/"$TEST \
                           	   		   -m $MODELS"/"$DATASET"_NLSE.pkl" \
                           	   		   -emb $FILTERED_EMBEDDINGS \
                               		   -run_id "NLSE" \
                           	   		   -res_path $RESULTS
									   -sub_size 5 \
									   -lrate 0.05 \
									   -n_epoch 5
fi