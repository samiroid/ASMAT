set -e
topic_datasets="duggan-1 duggan-3 cameron-1 flood-1"

#BOW
echo "" > "done_topics.txt"
for ds in $topic_datasets; 
do
    ./experiments/asma/scripts/BOW_ASMA.sh $ds topics.txt
    echo $ds >> "done_topics.txt"
done
# NEURAL

# 50-D vectors
for ds in $topic_datasets; 
do
    ./experiments/asma/scripts/neural_ASMA.sh $ds topics.txt str_skip_50.txt 50D
    echo $ds >> "done_topics.txt"
done

# 200-D vectors
for ds in $topic_datasets; 
do
    ./experiments/asma/scripts/neural_ASMA.sh $ds topics.txt str_skip_200.txt 200D
    echo $ds >> "done_topics.txt"
done

# 400-D vectors
for ds in $topic_datasets; 
do
    ./experiments/asma/scripts/neural_ASMA.sh $ds topics.txt str_skip_400.txt 400D
    echo $ds >> "done_topics.txt"
done