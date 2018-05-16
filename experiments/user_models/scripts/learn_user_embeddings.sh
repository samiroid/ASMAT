if [ -z "$1" ]
  then
    echo "please provide tweets dataset"
    exit 1
fi
DATASET=$1

if [ -z "$2" ]
  then
    echo "please provide output name"
    exit 1
fi
USER_EMBEDDINGS=$2

echo $RED"##### TRAIN USER EMBEDDINGS #####"$COLOR_OFF
# PROJECT_PATH="/Users/samir/Dev/projects/ASMAT/experiments/user_models/"
PROJECT_PATH="/data/ASMAT/ASMAT/experiments/user_models/"


WORKERS=10
NEGATIVE_SAMPLES=20
MIN_COUNT=5
VECTOR_DIM=200
# embeddings output
# EMBEDDINGS_OUT=$USER_EMBEDDINGS
# rm -rf $EMBEDDINGS_OUT
PV_EPOCHS=5
#$EMBEDDINGS_OUT"/PV-DM_"$VECTOR_DIM \
python ASMAT/toolkit/gensimer.py -input $PROJECT_PATH"/DATA/txt/"$DATASET \
								-output $PROJECT_PATH"/DATA/embeddings/"$USER_EMBEDDINGS"_"$VECTOR_DIM \
								-dim $VECTOR_DIM \
								-model "pv-dm" \
								-negative $NEGATIVE_SAMPLES \
								-min_count=$MIN_COUNT \
								-epochs $PV_EPOCHS \
								-workers $WORKERS 
								# -pretrained_vecs $WORD_EMBEDDINGS


