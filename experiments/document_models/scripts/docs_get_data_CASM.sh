set -e
#grab datasets from
DATASETS_SRC="/Users/samir/Dev/projects/ASMAT/RAW_DATA/raw_datasets/casm"
#place datasets at
DATASETS_DST="/Users/samir/Dev/projects/ASMAT/experiments/document_models/DATA/txt"

#####
# Add CASM datasets
####

CLEAN=0
GET_CASM_SENT=1
GET_CASM_TOPI=1

if (($CLEAN > 0)); then
	rm -rf $DATASETS_DST
fi
if (($GET_CASM_SENT > 0)); then
	#sentiment 
	sentiment_datasets="boo-cheer cameron-3 clacton clegg debate-1 farage miliband-1 miliband-2"
	for ds in $sentiment_datasets; 
	do
		echo reading $ds		

		python ASMAT/toolkit/dataset_parser.py -format "casm" \
										-input $DATASETS_SRC"/sentiment/"$ds"_test.txt" \
										$DATASETS_SRC"/sentiment/"$ds"_training.txt" \
										-outfolder $DATASETS_DST"/sentiment" -outname $ds".txt"		
		
		python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/sentiment/"$ds".txt" \
											 -train $DATASETS_DST"/"$ds"_train" \
											 -test $DATASETS_DST"/"$ds"_test" \
											 -dev $DATASETS_DST"/"$ds"_dev"											 
		###  SAME DATASETS WITH BINARY LABELS
		python ASMAT/toolkit/dataset_parser.py -format "casm" \
										-labels positive negative \
										-input $DATASETS_SRC"/sentiment/"$ds"_test.txt" \
										$DATASETS_SRC"/sentiment/"$ds"_training.txt" \
										-outfolder $DATASETS_DST"/sentiment" -outname $ds"~BINARY.txt"

		python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/sentiment/"$ds"~BINARY.txt" \
											 -train $DATASETS_DST"/"$ds"~BINARY_train" \
											 -test $DATASETS_DST"/"$ds"~BINARY_test" \
											 -dev $DATASETS_DST"/"$ds"~BINARY_dev"		
	done

fi	

if (($GET_CASM_TOPI > 0)); then
	#sentiment 
	topic_datasets="duggan-1 duggan-2 duggan-3 cameron-2 cameron-1 flood-1"
	for ds in $topic_datasets; 
	do
		echo reading $ds
		python ASMAT/toolkit/dataset_parser.py -format "casm" \
										-input $DATASETS_SRC"/topics/"$ds"_test.txt" \
										$DATASETS_SRC"/topics/"$ds"_training.txt" \
										-outfolder $DATASETS_DST"/topics" -outname $ds".txt"		
		
		python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/topics/"$ds".txt" \
											 -train $DATASETS_DST"/"$ds"_train" \
											 -test $DATASETS_DST"/"$ds"_test" \
											 -dev $DATASETS_DST"/"$ds"_dev"											 	
	done

fi	
