set -e
datasets="HCR OMD TW13 TW14 TW15"
# datasets="boo-cheer cameron-3"

for ds in $datasets; 
do
    ./experiments/low_resource/scripts/lexicon_LR.sh $ds"~BINARY_small" semlex "improving_lexicons_small.txt"
    ./experiments/low_resource/scripts/lexicon_LR.sh $ds"~BINARY_small" massive_lexicon "improving_lexicons_small.txt"
done


