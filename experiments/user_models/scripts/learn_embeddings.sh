#corpora paths
TWITTER_CORPUS="/Users/samir/Dev/resources/datasets/owoputi/raw_tweets.txt"
#TWITTER_CORPUS="/home/ubuntu/efs/work/datasets/owoputi_preprocessed.txt"
CORPUS="DATA/raw_datasets/mental_health/txt/corpus.txt"
USER_BOWS="DATA/raw_datasets/mental_health/txt/user_bows.txt"

WORD_EMBEDDINGS="DATA/embeddings/SG.txt"
#configs
WORKERS=2
NEGATIVE_SAMPLES=20
MIN_COUNT=10
VECTOR_DIM=50
# embeddings output
EMBEDDINGS_OUT="embeddings_"$VECTOR_DIM

# echo "#############################"
# echo " TRAIN PV"
# echo "#############################"
# PV_EPOCHS=2
# python ASMAT/toolkit/gensimer.py -input $USER_BOWS \
# 									-output $EMBEDDINGS_OUT"/PV-DM" \
# 									-dim $VECTOR_DIM \
# 									-model "pv-dm" \
# 									-negative $NEGATIVE_SAMPLES \
# 									-min_count=$MIN_COUNT \
# 									-epochs $PV_EPOCHS \
# 									-workers $WORKERS 
# exit

# python ASMAT/toolkit/gensimer.py -input $USER_BOWS \
# 									-output $EMBEDDINGS_OUT"/PV-DBOW" \
# 									-dim $VECTOR_DIM \
# 									-model "pv-dbow" \
# 									-negative $NEGATIVE_SAMPLES \
# 									-min_count=$MIN_COUNT \
# 									-epochs $PV_EPOCHS \
# 									-workers $WORKERS 
 									 
# echo "#############################"
# echo " TRAIN W2V"
# echo "#############################"

# python ASMAT/toolkit/gensimer.py -input $USER_BOWS $TWITTER_CORPUS \
# 									-out $EMBEDDINGS_OUT"/SG" \
# 									-dim $VECTOR_DIM \
# 									-model "skip" \
# 									-negative $NEGATIVE_SAMPLES \
# 									-min_count=$MIN_COUNT \
# 									-epochs 5 \
# 									-workers $WORKERS 

echo "#############################"
echo " USER 2 VEC "
echo "#############################"

#modules
U2V_PATH="/home/ubuntu/efs/work/projects/usr2vec/code"
U2V_PATH="/Users/samir/Dev/projects/usr2vec/code"
U2V_PATH="ASMAT/models/user2vec"

#u2v intermediate files
U2V_DATA="DATA/pkl/u2v_data.pkl"
U2V_DATA_AUX="DATA/pkl/u2v_data_aux.pkl"

### ACTION!
build=1
if (($build > 0 )); then
	# "DATA/txt/small_user_BOWS.txt"
	# -emb $EMBEDDINGS_OUT"/SG.txt" \
	printf "\n#### Build Training Data #####\n"	
	python $U2V_PATH"/build_train.py" -input $CORPUS \
								 	  -emb $WORD_EMBEDDINGS \
								 	  -output ${U2V_DATA} \
								 	  -min_word_freq $MIN_COUNT\
								 	  -neg_samples $NEGATIVE_SAMPLES	
fi
U2V_EPOCHS=5
printf "\n##### U2V training #####\n"
python $U2V_PATH"/train_u2v.py" -input ${U2V_DATA} \
								  -aux ${U2V_DATA_AUX} \
								  -output $EMBEDDINGS_OUT/"U2V" \
					 			  -patience 5 \
								  -margin 1 \
								  -epochs $U2V_EPOCHS \
								  -reshuff

exit



