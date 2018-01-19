set -e
#grab the lexicons from
LEXICONS_SRC="/Users/samir/Dev/resources/lexicons"
#place lexicons at
LEXICONS_DST="/Users/samir/Dev/projects/ASMAT/experiments/asma/input"
#grab datasets from
DATASETS_SRC="/Users/samir/Dev/projects/ASMAT/DATA/raw_datasets/casm"

#place datasets at
DATASETS_DST="/Users/samir/Dev/projects/ASMAT/experiments/asma/DATA/txt"
# DATA="/Users/samir/Dev/projects/ASMAT/experiments/asma/DATA"




#####
# Add CASM datasets
####

CLEAN=0
GET_CASM_SENT=1
GET_CASM_TOPI=0

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
		# cat $DATASETS_SRC_SENTIMENT"/"$ds"_test.txt"
		# cat $DATASETS_SRC_SENTIMENT"/"$ds"_training.txt"

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

	
	
	
	
# 	HCR=$DATASETS_SRC"/hcr"
# 	OMD=$DATASETS_SRC"/debate08"
# 	SEMEVAL=$DATASETS_SRC"/semeval_sentiment_raw"	
# 	# rm -rf $DATASETS_DST
# 	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
# 										-input $HCR"/dev.xml" $HCR"/test.xml" $HCR"/train.xml" \
# 										-outfolder $DATASETS_DST -outname "HCR.txt" 

# 	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
# 										-input $OMD"/dev.xml" $OMD"/test.xml" $OMD"/train.xml" \
# 										-outfolder $DATASETS_DST -outname "OMD.txt" 

# 	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
# 										-input $SEMEVAL"/Twitter2013_raw.txt" \
# 										-outfolder $DATASETS_DST -outname "TW13.txt" 
	
# 	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
# 										-input $SEMEVAL"/Twitter2014_raw.txt" \
# 										-outfolder $DATASETS_DST -outname "TW14.txt" 

# 	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
# 										-input $SEMEVAL"/Twitter2015_raw.txt" \
# 										-outfolder $DATASETS_DST -outname "TW15.txt" 
# fi



# #grab embeddings
# GET_EMBEDDINGS=1
# if (($GET_EMBEDDINGS > 0)); then
# 	mkdir -p "DATA/embeddings"	
# 	EMBEDDINGS_SRC="/Users/samir/Dev/resources/embeddings/twitter/str_skip/str_skip_50.txt"
# 	EMBEDDINGS_DST="DATA/embeddings/str_skip_50.txt"
# 	echo "embeddings: "$EMBEDDINGS_SRC" -> "$EMBEDDINGS_DST
# 	cp $EMBEDDINGS_SRC $EMBEDDINGS_DST
# fi