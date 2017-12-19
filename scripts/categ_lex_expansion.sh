set -e
# Reset
COLOR_OFF='\033[0m'       # Text Reset
# Regular Colors
RED='\033[0;31m'          # RED
if [ -z "$1" ]
  then
    echo "please provide lexicon"
    exit 1
fi
LEXICON=$1

if [ -z "$2" ]
  then
	RUN_ID=$LEXICON
else
	RUN_ID=$LEXICON"_"$2
fi

echo "LEXICON EXPANSION > " $LEXICON
INPUT="/Users/samir/Dev/projects/phd_research/low_resource_sma/DATA/lexicons/"$LEXICON".txt"
BASE="/Users/samir/Dev/projects/phd_research/low_resource_sma/experiments/lexicon_expansion/"$LEXICON"/"
#create folders
mkdir -p $BASE
DATA=$BASE"/DATA/"
FEATURES=$BASE"/features/"
MODELS=$BASE"/models/"
RESULTS=$BASE"/results/"

EMBEDDINGS="DATA/embeddings/str_skip_50.txt"
FILTERED_EMBEDDINGS=$FEATURES"vectors_str_skip_50.txt"

#TXT
TRAIN=$LEXICON"_train"
DEV=$LEXICON"_dev"
TEST=$LEXICON"_test"

CLEAN=0
if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $FEATURES/*.*
	rm -rf $MODELS/*.*
	rm -rf $DATA/*.*
	rm $RESULTS/*.*
fi

CLEAN_RESULTS=0
if (($CLEAN_RESULTS > 0)); then
	rm $RESULTS/*.*
fi

### DATA SPLIT ###
SPLIT=1
if (($SPLIT > 0)); then
	echo $RED"##### SPLIT DATA #####"$COLOR_OFF
	#first split the data into 80-20 split (temp/test)
	#then split the temp data into 80-20 split (train/dev)
	python ASMAT/toolkit/dataset_splitter.py -input $INPUT \
											 -output $DATA$LEXICON"_tmp" $DATA$LEXICON"_test" \
											 -rand_seed $RUN_ID \
											 -lexicon 
	python ASMAT/toolkit/dataset_splitter.py -input $DATA$LEXICON"_tmp" \
											 -output $DATA$LEXICON"_train" $DATA$LEXICON"_dev" \
											 -rand_seed $RUN_ID \
											 -lexicon 
	rm -rf $DATA$LEXICON"_tmp"
fi


### INDEX EXTRACTION ###
EXTRACT=1
if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX and EMBEDDINGS #####"$COLOR_OFF
	#extract vocabulary and indices
	python ASMAT/toolkit/extract.py -input $DATA$TRAIN $DATA$DEV $DATA$TEST \
									-out_folder $FEATURES \
									-vectors $EMBEDDINGS 	
fi

### COMPUTE FEATURES ###
GET_FEATURES=0
if (($GET_FEATURES > 0)); then
	echo $RED"##### GET FEATURES ##### "$COLOR_OFF
	#BOE
	python ASMAT/toolkit/features.py -input $FEATURES$TRAIN $FEATURES$DEV $FEATURES$TEST \
							-out_folder $FEATURES \
							-boe bin sum \
							-vectors $FILTERED_EMBEDDINGS	
	python ASMAT/toolkit/features.py -input $FEATURES$TRAIN \
							-out_folder $FEATURES \
							-nlse \
							-vectors $FILTERED_EMBEDDINGS	
fi

### LINEAR MODELS ###
LINEAR=0
if (($LINEAR > 0)); then
	VERBOSE=1
	echo $RED"##### LINEAR MODELS ##### "$COLOR_OFF
	python ASMAT/models/linear_model.py -verbose $VERBOSE -train $FEATURES$TRAIN \
							 -features BOE_bin -test $FEATURES$TEST \
							 -res_path $RESULTS"BOE.txt" 
	python ASMAT/models/linear_model.py -verbose $VERBOSE -train $FEATURES$TRAIN \
							 -features BOE_sum -test $FEATURES$TEST \
							 -res_path $RESULTS"BOE.txt"	
	python ASMAT/models/linear_model.py -verbose $VERBOSE -train $FEATURES$TRAIN \
							 -features BOE_bin BOE_sum -test $FEATURES$TEST \
							 -res_path $RESULTS"BOE.txt"	
fi

# ### NLSE #####
NLSE=1
if (($NLSE > 0)); then
	echo $RED"##### NLSE ##### "$COLOR_OFF
	python ASMAT/models/simple_nlse.py -tr $FEATURES$TRAIN"_NLSE.pkl" \
							   		   -dev $FEATURES$DEV \
                           	   		   -ts $FEATURES$TEST \
                           	   		   -m $MODELS$LEXICON"_NLSE.pkl" \
                           	   		   -emb $FILTERED_EMBEDDINGS \
                               		   -run_id "NLSE" \
                           	   		   -res_path $RESULTS$LEXICON"_NLSE.txt" \
									   -sub_size 5 \
									   -lrate 0.05 \
									   -n_epoch 5
fi