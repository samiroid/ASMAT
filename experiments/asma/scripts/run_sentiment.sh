set -e
sentiment_datasets="boo-cheer cameron-3 clacton clegg debate-1 farage miliband-1 miliband-2"
# sentiment_datasets="boo-cheer cameron-3"

echo "" > "done_sentiment.txt"
for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/supervised_ASMA.sh $ds sentiment.txt
    echo $ds >> "done_sentiment.txt"
done

echo "" > "done_sentiment_bin.txt"
for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/supervised_ASMA.sh $ds~BINARY sentiment~BINARY.txt
    echo $ds >> "done_sentiment_bin.txt"
done