set -e
datasets="HCR OMD TW13 TW14 TW15"
# datasets="boo-cheer cameron-3"

# for ds in $datasets; 
# do
#     ./experiments/low_resource/scripts/BOW_vs_Lex.sh $ds"~BINARY" "lexes_vs_bows.txt"
#     ./experiments/low_resource/scripts/best-lexicon_LR.sh $ds"~BINARY" semlex "lexes_vs_bows.txt"
#     ./experiments/low_resource/scripts/best-lexicon_LR.sh $ds"~BINARY" massive_lexicon "lexes_vs_bows.txt"    
#     ./experiments/low_resource/scripts/best-lexicon_LR.sh $ds"~BINARY" NRC_HS "lexes_vs_bows.txt"
#     ./experiments/low_resource/scripts/best-lexicon_LR.sh $ds"~BINARY" NRC_S140 "lexes_vs_bows.txt"    
# done


for ds in $datasets; 
do
    ./experiments/low_resource/scripts/BOW_vs_Lex.sh $ds"~BINARY_small" "lexes_vs_bows_small.txt"
    # ./experiments/low_resource/scripts/best-lexicon_LR.sh $ds"~BINARY_small" semlex "lexes_vs_bows_small.txt"
    # ./experiments/low_resource/scripts/best-lexicon_LR.sh $ds"~BINARY_small" massive_lexicon "lexes_vs_bows_small.txt"    
    # ./experiments/low_resource/scripts/best-lexicon_LR.sh $ds"~BINARY_small" NRC_HS "lexes_vs_bows_small.txt"
    # ./experiments/low_resource/scripts/best-lexicon_LR.sh $ds"~BINARY_small" NRC_S140 "lexes_vs_bows_small.txt"    
done

