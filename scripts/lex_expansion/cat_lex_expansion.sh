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
BASE=$PROJECT_PATH"/cat_lex_expansion/"$DATASET

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
EMBEDDINGS="DATA/embeddings/str_skip_50.txt"
FILTERED_EMBEDDINGS=$FEATURES"/vectors_str_skip_50.txt"

echo "LEXICON EXPANSION > " $DATASET

CLEAN=0
if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $FEATURES/*.* || True
	rm -rf $MODELS/*.* || True
	rm -rf $DATA/*.* || True
	rm $RESULTS/*.* || True
fi

### DATA SPLIT ###
SPLIT=0
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
EXTRACT=0
if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX #####"$COLOR_OFF
	#extract vocabulary and indices
	python ASMAT/toolkit/extract.py -input $DATA"/"$TRAIN $DATA"/"$DEV $DATA"/"$TEST -out_folder $FEATURES \
	-vectors $EMBEDDINGS 	
fi

### COMPUTE FEATURES ###
GET_FEATURES=0
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOE
	python ASMAT/toolkit/features.py -input $FEATURES"/"$TRAIN $FEATURES"/"$DEV $FEATURES"/"$TEST \
							-out_folder $FEATURES \
							-boe bin \
							-vectors $FILTERED_EMBEDDINGS	
	python ASMAT/toolkit/features.py -input $FEATURES"/"$TRAIN \
							-out_folder $FEATURES \
							-nlse \
							-vectors $FILTERED_EMBEDDINGS	
fi

### EXPANSION MODELS ###
EXPANDER=1
if (($EXPANDER > 0)); then
	VERBOSE=1
	echo $RED"##### EXPANSION MODELS ##### "$COLOR_OFF
	python ASMAT/apps/lexicon_expander.py -verbose $VERBOSE -train $FEATURES"/"$TRAIN \
										-test $FEATURES"/"$TEST \
										-features BOE_bin \
										-type categorical \
										-model linear \
										-res_path $RESULTS"/"$DATASET".txt"
	python ASMAT/apps/lexicon_expander.py -verbose $VERBOSE -train $FEATURES"/"$TRAIN \
										-test $FEATURES"/"$TEST \
										-features BOE_bin \
										-type categorical \
										-model l1 \
										-res_path $RESULTS"/"$DATASET".txt" 	
	python ASMAT/apps/lexicon_expander.py -verbose $VERBOSE -train $FEATURES"/"$TRAIN \
										-test $FEATURES"/"$TEST \
										-features BOE_bin \
										-type categorical \
										-model rbf \
										-res_path $RESULTS"/"$DATASET".txt"				

										
fi

# ### NLSE #####
NLSE=1
if (($NLSE > 0)); then
	echo $RED"##### NLSE ##### "$COLOR_OFF
	python ASMAT/models/train_nlse.py -tr $FEATURES"/"$TRAIN"_NLSE.pkl" \
							   		   -dev $FEATURES"/"$DEV \
                           	   		   -ts $FEATURES"/"$TEST \
                           	   		   -m $MODELS"/"$DATASET"_NLSE.pkl" \
                           	   		   -emb $FILTERED_EMBEDDINGS \
                               		   -run_id "NLSE" \
                           	   		   -res_path $RESULTS"/"$DATASET".txt" \
									   -sub_size 5 \
									   -lrate 0.05 \
									   -n_epoch 5
fi