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

# PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments"
# BASE=$PROJECT_PATH"/CV/neural_CV/"$DATASET


PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments"
BASE=$PROJECT_PATH"/asma/neural/"$DATASET
RESULTS=$PROJECT_PATH"/asma/results/"$DATASET".txt"
HYPERPARAMS=$PROJECT_PATH"/confs/default.cfg"

#create folders
mkdir -p $BASE
DATA=$BASE"/DATA"
FEATURES=$BASE"/features"
MODELS=$BASE"/models"
RESULTS=$BASE"/results"

#TXT
INPUT=$PROJECT_PATH"/datasets/"$DATASET".txt"
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"
EMBEDDINGS="DATA/embeddings/str_skip_50.txt"
FILTERED_EMBEDDINGS=$FEATURES"/str_skip_50.txt"

echo "NEURAL CV SMA > " $DATASET
#OPTIONS
CLEAN=1
SPLIT_CV=1
EXTRACT=1
GET_FEATURES=1
LINEAR=1
NLSE=1

CV=10

if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $BASE || True
fi

if (($SPLIT_CV > 0)); then
	echo $RED"##### CV DATA #####"$COLOR_OFF
	#cross-fold validation
	INPUT=$PROJECT_PATH"/datasets/"$DATASET".txt"
	python ASMAT/toolkit/dataset_splitter.py -input $INPUT \
											 -train $DATA"/"$DATASET"_train" \
											 -test $DATA"/"$DATASET"_test" \
											 -dev $DATA"/"$DATASET"_dev" \
											 -cv $CV \
											 -rand_seed $RUN_ID 
fi

if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX #####"$COLOR_OFF
	#extract vocabulary and indices
	python ASMAT/toolkit/extract.py -input $DATA"/"$TRAIN $DATA"/"$TEST $DATA"/"$DEV \
									-vocab_from $DATA"/"$TRAIN $DATA"/"$TEST $DATA"/"$DEV \
									-idx_labels \
									-out_folder $FEATURES \
									-cv $CV \
									-embeddings $EMBEDDINGS 
fi

### COMPUTE FEATURES ###
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOE
	python ASMAT/toolkit/features.py -input $FEATURES"/"$TRAIN $FEATURES"/"$TEST $FEATURES"/"$DEV \
							-out_folder $FEATURES \
							-boe bin sum \
							-cv $CV \
							-embeddings $FILTERED_EMBEDDINGS	
	# python ASMAT/toolkit/features.py -input $FEATURES"/"$TRAIN \
	# 						-out_folder $FEATURES \
	# 						-nlse \
	# 						-vectors $FILTERED_EMBEDDINGS	
fi

### LINEAR MODELS ###
if (($LINEAR > 0)); then
	VERBOSE=1
	echo $RED"##### LINEAR MODELS ##### "$COLOR_OFF
	python ASMAT/models/linear_model.py -train $FEATURES"/"$TRAIN \
							 -features BOE-BIN -test $FEATURES"/"$TEST \
							 -cv $CV \
							 -res_path $RESULTS
	python ASMAT/models/linear_model.py -train $FEATURES"/"$TRAIN \
							 -features BOE-SUM -test $FEATURES"/"$TEST \
							 -cv $CV \
							 -res_path $RESULTS	
fi

# ### NLSE #####
if (($NLSE > 0)); then
	echo $RED"##### NLSE ##### "$COLOR_OFF
	python ASMAT/models/train_nlse.py -tr $FEATURES"/"$TRAIN \
							   		   -dev $FEATURES"/"$DEV \
                           	   		   -ts $FEATURES"/"$TEST \
                           	   		   -m $MODELS"/"$DATASET"_NLSE.pkl" \
                           	   		   -emb_path $FILTERED_EMBEDDINGS \
									   -cv $CV \
                               		   -run_id "NLSE" \
                           	   		   -res_path $RESULTS \
									   -sub_size 5 \
									   -lrate 0.05 \
									   -n_epoch 10

	# python ASMAT/models/train_nlse.py -tr $FEATURES"/"$TRAIN"_NLSE.pkl" \
	# 						   		   -dev $FEATURES"/"$DEV \
    #                        	   		   -ts $FEATURES"/"$TEST \
    #                        	   		   -m $MODELS"/"$DATASET"_NLSE.pkl" \
    #                        	   		   -emb $FILTERED_EMBEDDINGS \
    #                            		   -run_id "NLSE" \
    #                        	   		   -res_path $RESULTS"/NLSE.txt" \
	# 								   -sub_size 5 \
	# 								   -lrate 0.05 \
	# 								   -n_epoch 5
fi