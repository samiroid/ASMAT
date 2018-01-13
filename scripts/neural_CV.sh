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
BASE=$PROJECT_PATH"/CV/neural_CV/"$DATASET

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
CLEAN=0
SPLIT=0
SPLIT_CV=0
EXTRACT=0
GET_FEATURES=0
LINEAR=1
NLSE=0

CV=10

if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $FEATURES/ || True
	rm -rf $MODELS/ || True
	rm -rf $DATA/ || True
	rm -rf $RESULTS/ || True
fi

### DATA SPLIT ###
if (($SPLIT > 0)); then
	echo $RED"##### SPLIT DATA #####"$COLOR_OFF
	#first split the data into 80-20 split (temp/test)
	#then split the temp data into 80-20 split (train/dev)
	
	python ASMAT/toolkit/dataset_splitter.py -input $INPUT \
											 -output $DATA"/"$DATASET"_tmp" $DATA"/"$DATASET"_test" \
											 -rand_seed $RUN_ID 
	python ASMAT/toolkit/dataset_splitter.py -input $DATA"/"$DATASET"_tmp" \
											 -output $DATA"/"$DATASET"_train" $DATA"/"$DATASET"_dev" \
											 -rand_seed $RUN_ID
	rm -rf $DATA"/"$DATASET"_tmp"
elif (($SPLIT_CV > 0)); then
	echo $RED"##### CV DATA #####"$COLOR_OFF
	#cross-fold validation
	INPUT=$PROJECT_PATH"/datasets/"$DATASET".txt"
	python ASMAT/toolkit/dataset_splitter.py -input $INPUT \
											 -output $DATA"/"$DATASET"_train" $DATA"/"$DATASET"_test" \
											 -cv $CV \
											 -rand_seed $RUN_ID
	# rm -rf $DATA"/"$DATASET"_tmp"
fi

if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX #####"$COLOR_OFF
	#extract vocabulary and indices
	python ASMAT/toolkit/extract.py -input $DATA"/"$TRAIN $DATA"/"$TEST \
								-vocab_from $DATA"/"$TRAIN $DATA"/"$TEST \
									-idx_labels \
									-out_folder $FEATURES \
									-cv $CV \
									-cv_from $DATA"/"$TRAIN $DATA"/"$TEST \
									-embeddings $EMBEDDINGS 
fi

### COMPUTE FEATURES ###
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOE
	python ASMAT/toolkit/features.py -input $FEATURES"/"$TRAIN $FEATURES"/"$TEST \
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
							 -features BOE_bin -test $FEATURES"/"$TEST \
							 -cv $CV \
							 -res_path $RESULTS"/BOE.txt" 
	# python ASMAT/models/linear_model.py -train $FEATURES"/"$TRAIN \
	# 						 -features BOE_sum -test $FEATURES"/"$TEST \
	# 						 -res_path $RESULTS"/BOE.txt"	
	# python ASMAT/models/linear_model.py -train $FEATURES"/"$TRAIN \
	# 						 -features BOE_bin BOE_sum -test $FEATURES"/"$TEST \
	# 						 -res_path $RESULTS"/BOE.txt"	
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
                           	   		   -res_path $RESULTS"/NLSE.txt" \
									   -sub_size 5 \
									   -lrate 0.05 \
									   -n_epoch 5
fi