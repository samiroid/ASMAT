set -e
sentiment_datasets="boo-cheer cameron-3 clacton clegg debate-1 farage miliband-1 miliband-2"
# sentiment_datasets="boo-cheer cameron-3"


echo "" > "done_sentiment.txt"

#BOW
for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/BOW_ASMA.sh $ds sentiment.txt
    echo $ds >> "done_sentiment.txt"
done

# NEURAL

# 50-D vectors
for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/neural_ASMA.sh $ds sentiment.txt str_skip_50.txt
    echo $ds >> "done_sentiment.txt"
done

# 200-D vectors
for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/neural_ASMA.sh $ds sentiment200.txt str_skip_200.txt
    echo $ds >> "done_sentiment.txt"
done

# 400-D vectors
for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/neural_ASMA.sh $ds sentiment400.txt str_skip_400.txt
    echo $ds >> "done_sentiment.txt"
done

