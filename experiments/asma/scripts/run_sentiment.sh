set -e
sentiment_datasets="boo-cheer cameron-3 clacton clegg debate-1 farage miliband-1 miliband-2"
# sentiment_datasets="boo-cheer cameron-3"

for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/supervised_ASMA.sh $ds sentiment.txt
    echo $ds >> "done.txt"
done

for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/supervised_ASMA.sh $ds~BINARY sentiment~BINARY.txt
    echo $ds >> "done.txt"
done