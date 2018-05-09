# if [ -z "$1" ]
#   then
#     echo "please provide tweets dataset"
#     exit 1
# fi
# DATASET=$1

# if [ -z "$2" ]
#   then
#     echo "please provide output name"
#     exit 1
# fi
# USER_EMBEDDINGS=$2

echo $RED"##### TRAIN WORD EMBEDDINGS #####"$COLOR_OFF
PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments/user_models/"
PROJECT_PATH="/data/ASMAT/ASMAT/experiments/user_models/"


WORKERS=10
NEGATIVE_SAMPLES=20
MIN_COUNT=10
VECTOR_DIM=400
# embeddings output
# EMBEDDINGS_OUT=$USER_EMBEDDINGS
# rm -rf $EMBEDDINGS_OUT
PV_EPOCHS=5
#$EMBEDDINGS_OUT"/PV-DM_"$VECTOR_DIM \
python ASMAT/toolkit/gensimer.py -input "RAW_DATA/raw_datasets/word_embeddings_corpus" "RAW_DATA/raw_datasets/all_tweets" \
								-output $PROJECT_PATH"/DATA/embeddings/"$USER_EMBEDDINGS \
								-dim $VECTOR_DIM \
								-model "skip" \
								-negative $NEGATIVE_SAMPLES \
								-min_count=$MIN_COUNT \
								-epochs $PV_EPOCHS \
								-workers $WORKERS 
								# -pretrained_vecs $WORD_EMBEDDINGS


