set -e
sentiment_datasets="HCR OMD TW13 TW14 TW15"
# sentiment_datasets="boo-cheer cameron-3"


echo "" > "done_lowresource.txt"

#BOW
for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/BOW_ASMA.sh $ds ""
    echo $ds >> "done_lowresource.txt"
done

# NEURAL

# 50-D vectors
for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/neural_ASMA.sh $ds "" str_skip_50.txt 50D
    echo $ds >> "done_lowresource.txt"
done

# 200-D vectors
for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/neural_ASMA.sh $ds "" str_skip_200.txt 200D
    echo $ds >> "done_lowresource.txt"
done

# 400-D vectors
for ds in $sentiment_datasets; 
do
    ./experiments/asma/scripts/neural_ASMA.sh $ds "" str_skip_400.txt 400D
    echo $ds >> "done_lowresource.txt"
done

