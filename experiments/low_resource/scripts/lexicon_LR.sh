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
BASE=$PROJECT_PATH"/sma/lexicon/"$DATASET
# BASE=$PROJECT_PATH"/lexicon_sma/"$DATASET
HYPERPARAMS=$PROJECT_PATH"/confs/lexicon.cfg"
CONFS=$PROJECT_PATH"/confs/default_lexicon.cfg"
LO_CONFS=$PROJECT_PATH"/confs/default_logodds_lexicon.cfg"

#create folders
mkdir -p $BASE
DATA=$BASE"/DATA"
#FEATURES=$BASE"/features"
#MODELS=$BASE"/models"
RESULTS=$BASE"/results"

#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"

LEXICON=$PROJECT_PATH"/lexicons/"$LEX_NAME
echo "LEXICON SMA > " $DATASET "@" $LEXICON

# OPTIONS
CLEAN=0
SPLIT=0
LOG_ODDS=1
LEX_SENT=0

if (($CLEAN > 0)); then
	echo "CLEAN-UP!"
	rm -rf $BASE || True
fi


### DATA SPLIT ###
if (($SPLIT > 0)); then
	echo $RED"##### SPLIT DATA #####"$COLOR_OFF	
	INPUT=$PROJECT_PATH"/datasets/"$DATASET".txt"
	python ASMAT/toolkit/dataset_splitter.py -input $INPUT \
											 -train $DATA"/"$TRAIN \
											 -test $DATA"/"$TEST \
											 -dev $DATA"/"$DEV \
											 -rand_seed $RUN_ID 
	
fi

if (($LEX_SENT > 0)); then
python ASMAT/toolkit/lexicon_classifier.py -lex $LEXICON".txt" -test_set $DATA"/"$TEST \
										-dev_set $DATA"/"$DEV \
									 	-res $RESULTS"/"$LEX_NAME".txt" \
									 	-out $RESULTS"/out_"$TEST".txt" \
										-confs_path $CONFS \
										-debug $BASE"/dbg/"

# python ASMAT/models/lexicon_sentiment.py -lex $LEXICON".txt" -test_set $DATA"/"$TEST \
# 										-dev_set $DATA"/"$DEV \
# 									 	-res $RESULTS"/"$LEX_NAME".txt" \
# 									 	-out $RESULTS"/out_"$TEST".txt" \
# 									 	-confs_path $CONFS \
# 									 	-hyperparams_path $HYPERPARAMS \		
fi 

if (($LOG_ODDS > 0)); then
	echo $RED"##### LOG ODDS CLASSIFIER #####"$COLOR_OFF	
python ASMAT/toolkit/lexicon_logodds.py -lex $LEXICON".txt" -test_set $DATA"/"$TEST \
										-dev_set $DATA"/"$DEV \
									 	-res $RESULTS"/"$LEX_NAME".txt" \
									 	-out $RESULTS"/out_"$TEST".txt" \
										-confs_path $LO_CONFS \
										-debug $BASE"/dbg/"
fi 

							 

