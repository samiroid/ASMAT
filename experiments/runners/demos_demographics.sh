set -e
VECTOR_DIM=400
RESULTS="demographics_"$VECTOR_DIM".tsv"
USER_EMBEDDINGS="demographics_u2v"
WORD_EMBEDDINGS="cohort_word_embeddings.txt"
#get datasets
./experiments/user_models/scripts/get_data_demographics.sh

#learn user embeddings > dataset embedding_filename
./experiments/user_models/scripts/learn_user_embeddings.sh $TWEETS $USER_EMBEDDINGS 

DATA="demos_age demos_gender demos_race"

for ds in $DATA; 
do
    #neurals > tweets labels word_embeddings user_embeddings results 
    ./experiments/user_models/scripts/users_neural.sh $TWEETS $ds $WORD_EMBEDDINGS  $USER_EMBEDDINGS"_"$VECTOR_DIM".txt" $RESULTS
    
done

for ds in $DATA; 
do
    #bows > tweets labels results 
    ./experiments/user_models/scripts/users_bow.sh $TWEETS $ds $RESULTS
done


