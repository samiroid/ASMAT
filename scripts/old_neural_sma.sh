if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi
DATASET=$1

echo "PROCESSING " $DATASET
BASE="/Users/samir/Dev/projects/phd_research/low_resource_sma/experiments/supervised_sma/neural/"$DATASET"/"

mkdir -p $BASE
RESULTS=$BASE"/results/"
PKL=$BASE"/pkl/"

#TXT
TRAIN_DATA="DATA/datasets/"$DATASET"_train.txt"
DEV_DATA="DATA/datasets/"$DATASET"_dev.txt"
TEST_DATA="DATA/datasets/"$DATASET"_test.txt"
#PKL
TRAIN_DATA_PKL=$PKL$DATASET"_train.pkl "
DEV_DATA_PKL=$PKL$DATASET"_dev.pkl"
TEST_DATA_PKL=$PKL$DATASET"_test.pkl"

EMBEDDINGS="DATA/embeddings/str_skip_50.txt"
FILTERED_EMBEDDINGS=$PKL"vectors_str_skip_50.txt"
CLEAN_PKL=0
if (($CLEAN_PKL > 0)); then
	rm -rf $PKL
fi

CLEAN_RESULTS=0
if (($CLEAN_RESULTS > 0)); then
	rm $RESULTS/*.*
fi

### INDEX EXTRACTION ###
EXTRACT=0
if (($EXTRACT > 0)); then
	echo "##### EXTRACT ##### "
	#extract vocabulary and indices
	python code/extract.py -input $TRAIN_DATA $DEV_DATA $TEST_DATA \
						   -out_folder $PKL \
						   -vectors $EMBEDDINGS 	
fi

### COMPUTE FEATURES ###
GET_FEATURES=0
if (($GET_FEATURES > 0)); then
	echo "##### GET FEATURES ##### "
	#BOE
	python code/features.py -input $TRAIN_DATA_PKL $DEV_DATA_PKL $TEST_DATA_PKL \
							-out_folder $PKL \
							-boe bin sum \
							-vectors $FILTERED_EMBEDDINGS	
	#NLSE
	python code/features.py -input $TRAIN_DATA_PKL \
							-out_folder $PKL \
							-nlse \
							-vectors $FILTERED_EMBEDDINGS
fi

### LINEAR MODELS ###
LINEAR=0
if (($LINEAR > 0)); then
	VERBOSE=1
	echo "##### LINEAR MODELS ##### "
	python code/doc_level.py -verbose $VERBOSE -train $TRAIN_DATA_PKL \
							 -features BOE_bin -test $TEST_DATA_PKL \
							 -res_path $RESULTS"BOE.txt" &&
	python code/doc_level.py -verbose $VERBOSE -train $TRAIN_DATA_PKL \
							 -features BOE_sum -test $TEST_DATA_PKL \
							 -res_path $RESULTS"BOE.txt"	
	python code/doc_level.py -verbose $VERBOSE -train $TRAIN_DATA_PKL \
							 -features BOE_sum BOE_bin -test $TEST_DATA_PKL \
							 -res_path $RESULTS"BOE.txt"	
fi

# ### NLSE #####
NLSE=1
if (($NLSE > 0)); then
	echo "##### NLSE ##### "
	python code/simple_nlse.py -tr $PKL$DATASET$"_train_NLSE.pkl" \
							   -dev $DEV_DATA_PKL \
                           	   -ts $TEST_DATA_PKL \
                           	   -m $PKL$DATASET"_NLSE_model.pkl" \
                           	   -emb $FILTERED_EMBEDDINGS \
                               -run_id "NLSE" \
                           	   -res_path $RESULTS"HCR_NLSE.txt" \
                           	   -sub_size 5 \
                           	   -lrate 0.05 \
                           	   -n_epoch 5
fi

