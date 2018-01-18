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
	echo "no RUN ID specified"
else
	RUN_ID=$DATASET"_"$2
fi

#config
PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments/asma"
DATA=$PROJECT_PATH"/DATA"
RESULTS=$DATA"/results/asma.txt"
LINEAR_FEATURES=$DATA"/pkl/linear_features"
NEURAL_FEATURES=$DATA"/pkl/neural_features"
MODELS=$DATA"/models"
#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"
EMB_FILE="str_skip_50.txt"
EMBEDDINGS="DATA/embeddings/"$EMB_FILE
FILTERED_EMBEDDINGS=$NEURAL_FEATURES"/"$DATASET"_"$EMB_FILE

LINEAR_HYPERPARAMS=$PROJECT_PATH"/confs/linear.cfg"
NLSE_HYPERPARAMS=$PROJECT_PATH"/confs/nlse.cfg"

echo "LINEAR SMA > " $DATASET
#OPTIONS
CLEAN=1
EXTRACT=1
GET_FEATURES=1
LINEAR_MODELS=1
NLSE=1

if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $PROJECT_PATH"/pkl" || True
fi


# if (($SPLIT > 0)); then
# 	echo $RED"##### SPLIT DATA #####"$COLOR_OFF
	
# 	INPUT=$PROJECT_PATH"/input/"$DATASET".txt"
# 	python ASMAT/toolkit/dataset_splitter.py -input $INPUT \
# 											 -train $DATA"/"$TRAIN \
# 											 -test $DATA"/"$TEST \
# 											 -dev $DATA"/"$DEV \
# 											 -rand_seed $RUN_ID 
	
# fi

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
	#NOTE (cont.): whereas embedding based models can represent all words in the embedding matrix
	python ASMAT/toolkit/extract.py -input $DATA"/txt/"$TRAIN $DATA"/txt/"$DEV \
										   $DATA"/txt/"$TEST \
									-vocab_from $DATA"/txt/"$TRAIN $DATA"/txt/"$DEV \
												$DATA"/txt/"$TEST \
									-out_folder $NEURAL_FEATURES \
									-idx_labels	\
									-embeddings $EMBEDDINGS 
	mv $NEURAL_FEATURES"/"$EMB_FILE $FILTERED_EMBEDDINGS

fi
### COMPUTE FEATURES ###
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOW
	python ASMAT/toolkit/features.py -input $LINEAR_FEATURES"/"$TRAIN $LINEAR_FEATURES"/"$DEV $LINEAR_FEATURES"/"$TEST \
									-bow bin \
									-out_folder $LINEAR_FEATURES 
	
	python ASMAT/toolkit/features.py -input $NEURAL_FEATURES"/"$TRAIN $NEURAL_FEATURES"/"$DEV $NEURAL_FEATURES"/"$TEST \
							-out_folder $NEURAL_FEATURES \
							-boe bin sum \
							-embeddings $FILTERED_EMBEDDINGS	
fi

### LINEAR MODELS ###
if (($LINEAR_MODELS > 0)); then
	echo $RED"##### LINEAR MODELS ##### "$COLOR_OFF	
	
	python ASMAT/toolkit/linear_model.py -features BOW-BIN \
										-train $LINEAR_FEATURES"/"$TRAIN \
							  			-test $LINEAR_FEATURES"/"$TEST \
										-dev $LINEAR_FEATURES"/"$DEV \
							 			-res_path $RESULTS 
										

	python ASMAT/toolkit/linear_model.py -features naive_bayes \
										-train $LINEAR_FEATURES"/"$TRAIN \
							  			-test $LINEAR_FEATURES"/"$TEST \
										-dev $LINEAR_FEATURES"/"$DEV \
							 			-res_path $RESULTS

	python ASMAT/toolkit/linear_model.py -train $NEURAL_FEATURES"/"$TRAIN \
								-dev $NEURAL_FEATURES"/"$DEV \
							 -features BOE-BIN -test $NEURAL_FEATURES"/"$TEST \
							 -res_path $RESULTS 
							 
	python ASMAT/toolkit/linear_model.py -train $NEURAL_FEATURES"/"$TRAIN -dev $NEURAL_FEATURES"/"$DEV \
							 -features BOE-SUM -test $NEURAL_FEATURES"/"$TEST \
							 -res_path $RESULTS 
	
fi


if (($NLSE > 0)); then
	echo $RED"##### NLSE ##### "$COLOR_OFF
	python ASMAT/toolkit/train_nlse.py -tr $NEURAL_FEATURES"/"$TRAIN \
							   		   -dev $NEURAL_FEATURES"/"$DEV \
                           	   		   -ts $NEURAL_FEATURES"/"$TEST \
                           	   		   -m $MODELS"/"$DATASET"_NLSE.pkl" \
                           	   		   -emb $FILTERED_EMBEDDINGS \
                               		   -run_id "NLSE" \
                           	   		   -res_path $RESULTS \
									   -sub_size 5 \
									   -lrate 0.05 \
									   -n_epoch 5 
fi