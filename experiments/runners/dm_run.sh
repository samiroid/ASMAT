set -e
sentiment_datasets="HCR OMD TW13 TW14 TW15"
# sentiment_datasets="boo-cheer cameron-3"


echo "" > "done_dm.txt"

#BOW
# for ds in $sentiment_datasets; 
# do
#     ./experiments/document_models/scripts/dm_BOW.sh $ds ""
#     echo $ds >> "done_dm.txt"
# done

# NEURAL

# 50-D vectors
for ds in $sentiment_datasets; 
do
    ./experiments/document_models/scripts/dm_neural.sh $ds "" str_skip_50.txt 50D
    echo $ds >> "done_dm.txt"
done

# # 200-D vectors
# for ds in $sentiment_datasets; 
# do
#     ./experiments/document_models/scripts/dm_neural.sh $ds "" str_skip_200.txt 200D
#     echo $ds >> "done_dm.txt"
# done

# # 400-D vectors
# for ds in $sentiment_datasets; 
# do
#     ./experiments/document_models/scripts/dm_neural.sh $ds "" str_skip_400.txt 400D
#     echo $ds >> "done_dm.txt"
done

