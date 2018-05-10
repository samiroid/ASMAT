set -e
VECTOR_DIM=400
#get datasets
#./experiments/user_models/scripts/get_data_mentalhealth.sh

#learn user embeddings
#./experiments/user_models/scripts/learn_user_embeddings.sh mental_health_tweets mental_health_u2v

#bows > tweets labels results 
./experiments/user_models/scripts/users_bow.sh mental_health_tweets mental_health "demos_mental_health_"$VECTOR_DIM".txt"

#neurals > tweets labels word_embeddings user_embeddings results 
./experiments/user_models/scripts/users_neural.sh mental_health_tweets mental_health cohort_word_embeddings.txt "mental_health_u2v_"$VECTOR_DIM".txt" "demos_mental_health_"$VECTOR_DIM".txt"
