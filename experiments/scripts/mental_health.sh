set -e
VECTOR_DIM=200
TWEETS="mental_health_tweets"
USER_EMBEDDINGS="mental_health_u2v_"$VECTOR_DIM".txt" 
RESULTS="mental_health_"$VECTOR_DIM".tsv"
WORD_EMBEDDINGS="cohort_word_embeddings.txt"
# WORD_EMBEDDINGS="str_skip_50.txt"

# #get datasets
./experiments/user_models/scripts/get_data_mentalhealth.sh

# #learn user embeddings
./experiments/user_models/scripts/learn_user_embeddings.sh $TWEETS $USER_EMBEDDINGS 

DATA="depression ptsd"
for ds in $DATA; 
do
    #bows > tweets labels results 
    ./experiments/user_models/scripts/users_bow.sh $TWEETS $ds $RESULTS    
done

for ds in $DATA; 
do
    #neurals > tweets labels word_embeddings user_embeddings results 
    ./experiments/user_models/scripts/users_neural.sh $TWEETS $ds $WORD_EMBEDDINGS  $USER_EMBEDDINGS $RESULTS
done


