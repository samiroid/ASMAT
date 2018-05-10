set -e
VECTOR_DIM=400

#get datasets
./experiments/user_models/scripts/get_data_demographics.sh

#learn user embeddings > dataset embedding_filename
./experiments/user_models/scripts/learn_user_embeddings.sh demos_tweets demographics_u2v

DATA="demos_age demos_gender demos_race"

for ds in $DATA; 
do
    #neurals > tweets labels word_embeddings user_embeddings results 
    ./experiments/user_models/scripts/users_neural.sh demos_tweets $ds cohort_word_embeddings.txt "demographics_u2v_"$VECTOR_DIM".txt" "demographics_"$VECTOR_DIM".txt"
done

for ds in $DATA; 
do
    #bows > tweets labels results 
    ./experiments/user_models/scripts/users_bow.sh demos_tweets $ds "demographics_"$VECTOR_DIM".txt"
done


