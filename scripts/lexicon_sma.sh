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
    echo "please provide lexicon"
    exit 1
fi
LEX_NAME=$2
if [ -z "$3" ]
  then
	RUN_ID=$DATASET
else
	RUN_ID=$DATASET"_"$3
fi

PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments"
BASE=$PROJECT_PATH"/lexicon_sma/"$DATASET

#create folders
mkdir -p $BASE
DATA=$BASE"/DATA"
#FEATURES=$BASE"/features"
MODELS=$BASE"/models"
RESULTS=$BASE"/results"

#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"


# LEXICONS_PATH="/Users/samir/Dev/projects/phd_research/low_resource_sma/DATA/lexicons/"
# LEXICON=$LEXICONS_PATH$LEX_NAME

LEXICON=$PROJECT_PATH"/lexicons/"$LEX_NAME
echo "LEXICON SMA > " $DATASET "@" $LEXICON



CLEAN=0
if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
#	rm -rf $FEATURES/*.* || True
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


python ASMAT/models/lexicon_model.py -lex $LEXICON".txt" -ts $DATA"/"$TEST -res $RESULTS"/"$LEX_NAME".txt" 

