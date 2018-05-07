
echo $RED"##### TRAIN USER EMBEDDINGS #####"$COLOR_OFF
WORKERS=2
NEGATIVE_SAMPLES=20
MIN_COUNT=10
VECTOR_DIM=50
# embeddings output
# EMBEDDINGS_OUT=$USER_EMBEDDINGS
# rm -rf $EMBEDDINGS_OUT
PV_EPOCHS=5
#$EMBEDDINGS_OUT"/PV-DM_"$VECTOR_DIM \
python ASMAT/toolkit/gensimer.py -input $DATA"/txt/"$TWEETS \
								-output $USER_EMBEDDINGS \
								-dim $VECTOR_DIM \
								-model "pv-dm" \
								-negative $NEGATIVE_SAMPLES \
								-min_count=$MIN_COUNT \
								-epochs $PV_EPOCHS \
								-workers $WORKERS 
								# -pretrained_vecs $WORD_EMBEDDINGS


