set -e
sentiment_datasets="HCR OMD TW13 TW14 TW15"


echo "" > "done_dm.txt"



# NEURAL
#for ds in $sentiment_datasets; 
#do
#    ./experiments/document_models/scripts/docs_neural.sh $ds "" cohort_w2v_400.txt "400"
#    echo $ds >> "done_dm.txt"
#done

#BOW
for ds in $sentiment_datasets; 
do
    ./experiments/document_models/scripts/docs_BOW.sh $ds ""
    echo $ds >> "done_dm.txt"
done


