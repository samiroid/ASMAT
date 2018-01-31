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
	RESFILE="asma.txt"
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

PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments/asma"
DATA=$PROJECT_PATH"/DATA"
RESULTS=$DATA"/results/"$RESFILE


HYPERPARAMS=$PROJECT_PATH"/confs/lexicon.cfg"
CONFS=$PROJECT_PATH"/confs/default_lexicon.cfg"



#TXT
TRAIN=$DATASET"_train"
DEV=$DATASET"_dev"
TEST=$DATASET"_test"

LEXICON="DATA/lexicons/"$LEX_NAME
echo "LEXICON SMA > " $DATASET "@" $LEX_NAME

# OPTIONS
CLEAN=0
LEX_SENT=1
if (($LEX_SENT > 0)); then
python ASMAT/toolkit/lexicon_classifier.py -lex $LEXICON".txt" \
											-run_id $RUN_ID"_t" \
											-test_set $DATA"/txt/"$TEST \
											-dev_set $DATA"/txt/"$DEV \
									 		-res $RESULTS \
											-confs_path $CONFS \
											-hyperparams_path "" \
											-debug "dbg/"

python ASMAT/toolkit/lexicon_classifier_bkp.py -lex $LEXICON".txt" \
											-run_id $RUN_ID \
											-test_set $DATA"/txt/"$TEST \
											-dev_set $DATA"/txt/"$DEV \
									 		-res $RESULTS \
											-confs_path $CONFS \
											-hyperparams_path $HYPERPARAMS \
											-debug "dbg/"

# python ASMAT/models/lexicon_sentiment.py -lex $LEXICON".txt" -test_set $DATA"/"$TEST \
# 										-dev_set $DATA"/"$DEV \
# 									 	-res $RESULTS"/"$LEX_NAME".txt" \
# 									 	-out $RESULTS"/out_"$TEST".txt" \
# 									 	-confs_path $CONFS \
# 									 	-hyperparams_path $HYPERPARAMS \		
fi 


							 

