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
    echo "please provide word embeddings file"
    exit 1
fi
WORD_EMBS=$3

if [ -z "$4" ]
  then
    echo "please provide user embeddings file"
    exit 1
fi

USER_EMBS=$4
if [ -z "$5" ]
  then
	RESFILE="user_models.txt"
	echo "default results file: " $RESFILE
else
	RESFILE=$5
fi

if [ -z "$7" ]
  then
	RUN_ID=$WORD_EMBS
	echo "default RUN ID"
else
	RUN_ID=$7
fi
#config
PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments/user_models"
DATA=$PROJECT_PATH"/DATA"
RESULTS=$DATA"/results/"$RESFILE
NEURAL_FEATURES=$DATA"/pkl/neural_features"
MODELS=$DATA"/models"
#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"
#TWEETS=$DATASET"_users_tweets"
#USER_EMBEDDINGS=$DATA/"embeddings/U2V"
#USER_EMBEDDINGS=$NEURAL_FEATURES"/embeddings/U2V_"$DATASET
#WORD_EMBEDDINGS_INPUT="RAW_DATA/embeddings/"$EMB_FILE

USER_EMBEDDINGS=$PROJECT_PATH"/DATA/embeddings/"$USER_EMBS
WORD_EMBEDDINGS_INPUT="RAW_DATA/embeddings/"$WORD_EMBS
WORD_EMBEDDINGS=$NEURAL_FEATURES"/"$DATASET"_"$WORD_EMBS

NLSE_HYPERPARAMS=$PROJECT_PATH"/confs/nlse.cfg"
LINEAR_HYPERPARAMS=$PROJECT_PATH"/confs/linear.cfg"

echo "word embeddingd @" $WORD_EMBEDDINGS_INPUT
echo "user embeddingd @" $USER_EMBEDDINGS

echo "NEURAL SMA > " $DATASET
#OPTIONS
CLEAN=0
EXTRACT=0
GET_FEATURES=0
LINEAR_MODELS=0
NLSE=1
HYPERPARAM=0
if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $DATA"/pkl" || True
fi

if [[ $HYPERPARAM -eq 0 ]]; then
	NLSE_HYPERPARAMS="none"
	LINEAR_HYPERPARAMS="none"  
fi

### INDEX EXTRACTION ###
if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX #####"$COLOR_OFF
	#extract vocabulary and indices	
	#NOTE embedding based models can represent all words in the embedding matrix so it is 
	# ok to include the test set in the vocabulary
	python $PROJECT_PATH/code/users_extract.py -labels_path $DATA"/txt/"$TRAIN $DATA"/txt/"$DEV \
										   $DATA"/txt/"$TEST \
										   -text_path $DATA"/txt/"$TWEETS \
									-vocab_from $DATA"/txt/"$TRAIN $DATA"/txt/"$DEV \
												$DATA"/txt/"$TEST \
									-idx_labels \
									-out_folder $NEURAL_FEATURES \
									-embeddings $WORD_EMBEDDINGS_INPUT 
	#word embeddings file only with the words on this vocabulary
	mv $NEURAL_FEATURES"/"$WORD_EMBS $WORD_EMBEDDINGS
fi

### COMPUTE FEATURES ###
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOE	
	python ASMAT/toolkit/features.py -input $NEURAL_FEATURES"/"$TRAIN $NEURAL_FEATURES"/"$DEV $NEURAL_FEATURES"/"$TEST \
							-out_folder $NEURAL_FEATURES \
							-boe bin sum \
							-embeddings $WORD_EMBEDDINGS	
	
	python ASMAT/toolkit/features.py -input $NEURAL_FEATURES"/users_"$TRAIN $NEURAL_FEATURES"/users_"$DEV $NEURAL_FEATURES"/users_"$TEST \
							-out_folder $NEURAL_FEATURES \
							-boe bin \
							-embeddings $USER_EMBEDDINGS
fi

### LINEAR MODELS ###
if (($LINEAR_MODELS > 0)); then
	echo $RED"##### LINEAR MODELS ##### "$COLOR_OFF	
	#USER-LEVEL
	python ASMAT/toolkit/linear_model.py -features BOE-BIN \
										-run_id $RUN_ID \
										-train $NEURAL_FEATURES"/users_"$TRAIN \
										-test $NEURAL_FEATURES"/users_"$TEST \
										-dev $NEURAL_FEATURES"/users_"$DEV \
							 			-res_path $RESULTS \
										-hyperparams_path $LINEAR_HYPERPARAMS
										
	python ASMAT/toolkit/linear_model.py -features BOE-BIN \
										-run_id $RUN_ID \
										-train $NEURAL_FEATURES"/"$TRAIN \
										-test $NEURAL_FEATURES"/"$TEST \
										-dev $NEURAL_FEATURES"/"$DEV \
							 			-res_path $RESULTS \
										-hyperparams_path $LINEAR_HYPERPARAMS
							 
	python ASMAT/toolkit/linear_model.py -features BOE-SUM \
										-run_id $RUN_ID \
										-train $NEURAL_FEATURES"/"$TRAIN \
										-test $NEURAL_FEATURES"/"$TEST \
										-dev $NEURAL_FEATURES"/"$DEV \
										-res_path $RESULTS \
										-hyperparams_path $LINEAR_HYPERPARAMS
	
fi


if (($NLSE > 0)); then
	echo $RED"##### NLSE ##### "$COLOR_OFF
	python ASMAT/toolkit/train_nlse.py -train $NEURAL_FEATURES"/users_"$TRAIN \
							   		   -dev $NEURAL_FEATURES"/users_"$DEV \
                           	   		   -test $NEURAL_FEATURES"/users_"$TEST \
                           	   		   -m $MODELS"/"$DATASET"_NLSE.pkl" \
                           	   		   -emb $USER_EMBEDDINGS \
                               		   -run_id $RUN_ID\
                           	   		   -res_path $RESULTS \
									   -sub_size 10 \
									   -lrate 0.01 \
									   -n_epoch 20 \
									   -patience 20 
									#    \
									#    -hyperparams_path $NLSE_HYPERPARAMS

	# python ASMAT/toolkit/train_nlse_2.py -train $NEURAL_FEATURES"/"$TRAIN \
	# 						   		   -dev $NEURAL_FEATURES"/"$DEV \
    #                        	   		   -test $NEURAL_FEATURES"/"$TEST \
    #                        	   		   -m $MODELS"/"$DATASET"_NLSE.pkl" \
    #                        	   		   -emb $FILTERED_EMBEDDINGS \
    #                            		   -run_id $RUN_ID"_2" \
    #                        	   		   -res_path $RESULTS \
	# 								   -sub_size 5 \
	# 								   -lrate 0.05 \
	# 								   -n_epoch 20 \
	# 								   -patience 8 \
	# 								   -hyperparams_path $NLSE_HYPERPARAMS
fi