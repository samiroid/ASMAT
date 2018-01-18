set -e
#grab the lexicons from
LEXICONS_SRC="/Users/samir/Dev/resources/lexicons"
#place lexicons at
LEXICONS_DST="/Users/samir/Dev/projects/ASMAT/experiments/low_resource/input"
#grab datasets from
DATASETS_SRC="/Users/samir/Dev/projects/ASMAT/DATA/raw_datasets"
#place datasets at
DATASETS_DST="/Users/samir/Dev/projects/ASMAT/experiments/low_resource/input"



#####
# Add datasets
####

GET_DATASETS=1
if (($GET_DATASETS > 0)); then
	HCR=$DATASETS_SRC"/hcr"
	OMD=$DATASETS_SRC"/debate08"
	SEMEVAL=$DATASETS_SRC"/semeval_sentiment_raw"	
	rm -rf $DATASETS_DST
	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-input $HCR"/dev.xml" $HCR"/test.xml" $HCR"/train.xml" \
										-outfolder $DATASETS_DST -outname "HCR.txt" 

	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-input $OMD"/dev.xml" $OMD"/test.xml" $OMD"/train.xml" \
										-outfolder $DATASETS_DST -outname "OMD.txt" 

	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-input $SEMEVAL"/Twitter2013_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW13.txt" 
	
	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-input $SEMEVAL"/Twitter2014_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW14.txt" 

	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-input $SEMEVAL"/Twitter2015_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW15.txt" 
	

fi

# #grab embeddings
# GET_EMBEDDINGS=1
# if (($GET_EMBEDDINGS > 0)); then
# 	mkdir -p "DATA/embeddings"	
# 	EMBEDDINGS_SRC="/Users/samir/Dev/resources/embeddings/twitter/str_skip/str_skip_50.txt"
# 	EMBEDDINGS_DST="DATA/embeddings/str_skip_50.txt"
# 	echo "embeddings: "$EMBEDDINGS_SRC" -> "$EMBEDDINGS_DST
# 	cp $EMBEDDINGS_SRC $EMBEDDINGS_DST
# fi