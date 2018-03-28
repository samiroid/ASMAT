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
	RESFILE="lexicons.txt"
	echo "default results file"
else
	RESFILE=$3
fi

if [ -z "$4" ]
  then
	RUN_ID=$LEX_NAME
else
	RUN_ID=$4
fi

PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments/lexicon_models"
DATA=$PROJECT_PATH"/DATA"
RESULTS=$DATA"/results/"$RESFILE
HYPERPARAMS=$PROJECT_PATH"/confs/hyper-lexicon.cfg"
CONFS=$PROJECT_PATH"/confs/default_lexicon.cfg"

#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"

LEXICON="DATA/lexicons/"$LEX_NAME
echo "LEXICON SMA > " $DATASET "@" $LEX_NAME

# OPTIONS
CLEAN=0
NO_FIT=1
FIT=1
TUNE=1
if (($CLEAN > 0)); then
	rm -rf $RESULTS
fi
if (($NO_FIT > 0)); then
echo $RED"##### NO FIT ##### "$COLOR_OFF	
python ASMAT/toolkit/lexicon_classifier.py -lex $LEXICON".txt" \
											-run_id $RUN_ID \
											-test_set $DATA"/txt/"$TEST \
									 		-res $RESULTS \
											-confs_path $CONFS											
fi 

if (($FIT > 0)); then
echo $RED"##### FIT ##### "$COLOR_OFF	
python ASMAT/toolkit/lexicon_classifier.py -lex $LEXICON".txt" \
											-run_id $RUN_ID"-fitted" \
											-test_set $DATA"/txt/"$TEST \
											-dev_set $DATA"/txt/"$TRAIN \
									 		-res $RESULTS \
											-confs_path $CONFS											
fi 

if (($TUNE > 0)); then
echo $RED"##### FIT AND TUNE ##### "$COLOR_OFF	
python ASMAT/toolkit/lexicon_classifier.py -lex $LEXICON".txt" \
											-run_id $RUN_ID"-tunned" \
											-test_set $DATA"/txt/"$TEST \
											-dev_set $DATA"/txt/"$TRAIN \
									 		-res $RESULTS \
											-confs_path $CONFS \
											-hyperparams_path $HYPERPARAMS										
fi 
