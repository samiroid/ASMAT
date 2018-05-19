set -e
VECTOR_DIM=200
USER_EMBEDDINGS="cohort_u2v"
DATA_PATH="/data/ASMAT/ASMAT/experiments/demos/DATA"
#DATA_PATH="/Users/samir/Dev/projects/ASMAT/experiments/demos/DATA"
NEURAL_FEATURES=$DATA_PATH"/pkl/features"

#RUN CONFIGS
GET_DATA=0
EXTRACT=0
TRAIN_U2V=0
RUN_MH=1
RUN_DEMO=1
FEATURES=0
LINEAR=0
NLSE=1
NLSE_INFER=0
NLSE_HYPERPARAMS=$DATA_PATH"/confs/nlse.cfg"

#get datasets
if (($GET_DATA > 0)); then
    echo $RED"##### GET DATA #####"$COLOR_OFF
    ./experiments/demos/scripts/get_data_demographics.sh
    ./experiments/demos/scripts/get_data_mentalhealth.sh
fi

if (($TRAIN_U2V > 0)); then
    #learn user embeddings > dataset embedding_filename
    echo $RED"##### TRAIN USER EMBEDDINGS #####"$COLOR_OFF
    WORKERS=4
    NEGATIVE_SAMPLES=10
    MIN_COUNT=5    
    PV_EPOCHS=5
    python ASMAT/toolkit/gensimer.py -input $DATA_PATH"/txt/demos_tweets" \
                                            $DATA_PATH"/txt/mental_health_tweets" \
                                    -output $DATA_PATH"/embeddings/"$USER_EMBEDDINGS \
                                    -dim $VECTOR_DIM \
                                    -model "pv-dm" \
                                    -negative $NEGATIVE_SAMPLES \
                                    -min_count=$MIN_COUNT \
                                    -epochs $PV_EPOCHS \
                                    -workers $WORKERS 
fi

if (($EXTRACT > 0)); then
	echo $RED"##### EXTRACT INDEX #####"$COLOR_OFF
	#extract vocabulary and indices	
	#NOTE embedding based models can represent all words in the embedding matrix so it is 
	# ok to include the test set in the vocabulary
    DATASETS="$DATA_PATH/txt/age_train $DATA_PATH/txt/age_test $DATA_PATH/txt/age_dev $DATA_PATH/txt/gender_train $DATA_PATH/txt/gender_test $DATA_PATH/txt/gender_dev $DATA_PATH/txt/race_train $DATA_PATH/txt/race_test $DATA_PATH/txt/race_dev $DATA_PATH/txt/ptsd_train $DATA_PATH/txt/ptsd_test $DATA_PATH/txt/ptsd_dev $DATA_PATH/txt/depression_train $DATA_PATH/txt/depression_test $DATA_PATH/txt/depression_dev $DATA_PATH/txt/cohort.txt"

    python ASMAT/toolkit/users_extract.py -users_only -labels_path $DATASETS \
									            -out_folder $NEURAL_FEATURES
fi

MH_DATASETS="depression ptsd"
DEMO_DATASETS="age race gender"
DATASETS=""
if (($RUN_MH > 0)); then
    for ds in $MH_DATASETS; do
        DATASETS=$DATASETS" "$ds        
    done
fi

if (($RUN_DEMO > 0)); then
    for ds in $DEMO_DATASETS; do
        DATASETS=$DATASETS" "$ds       
    done
fi

if (($FEATURES > 0)); then
    for ds in $DATASETS; do
        echo $RED"##### GET FEATURES $ds ##### "$COLOR_OFF		
	    python ASMAT/toolkit/features.py -input $NEURAL_FEATURES"/"$ds"_train_users" \
                                            $NEURAL_FEATURES"/"$ds"_dev_users" \
                                            $NEURAL_FEATURES"/"$ds"_test_users" \
							-out_folder $NEURAL_FEATURES \
							-u2v \
							-embeddings $DATA_PATH"/embeddings/"$USER_EMBEDDINGS"_"$VECTOR_DIM".txt"
    done
fi

if (($LINEAR > 0)); then    
    for ds in $DATASETS; do        
        echo $RED"##### LINEAR MODEL $ds ##### "$COLOR_OFF	
        python ASMAT/toolkit/linear_model.py -features u2v \
                                            -run_id "DEMOS" \
                                            -train $NEURAL_FEATURES"/"$ds"_train_users" \
                                            -test $NEURAL_FEATURES"/"$ds"_test_users" \
                                            -dev $NEURAL_FEATURES"/"$ds"_dev_users" 
                                            #-hyperparams_path $LINEAR_HYPERPARAMS
 done
 fi

if (($NLSE > 0)); then    
    for ds in $DATASETS; do        
        echo $RED"##### NLSE MODEL $ds ##### "$COLOR_OFF	
        python ASMAT/toolkit/train_nlse_2.py -run_id "DEMOS" \
                                            -train $NEURAL_FEATURES"/"$ds"_train_users" \
                                            -test $NEURAL_FEATURES"/"$ds"_test_users" \
                                            -dev $NEURAL_FEATURES"/"$ds"_dev_users" \
                           	   		   -m $DATA_PATH"/models/"$ds"_NLSE.pkl" \
                           	   		   -emb $DATA_PATH"/embeddings/"$USER_EMBEDDINGS"_"$VECTOR_DIM".txt" \
									   -sub_size 5 \
									   -lrate 0.01 \
									   -n_epoch 20 \
									   -patience 10 \
                                       -hyperparams_path $NLSE_HYPERPARAMS
 done
 fi

if (($NLSE_INFER > 0)); then
    for ds in $DATASETS; do        
	    echo $MAGENTA"##### NLSE INFERENCE $ds ##### "$COLOR_OFF
	    python ASMAT/toolkit/run_nlse.py -data_path $DATA_PATH"/txt/cohort.txt" \
								 	-model_path $DATA_PATH"/models/"$ds"_NLSE.pkl" \
									-res_path $DATA_PATH"/results/predictions_"$ds".txt"			
    done
fi
