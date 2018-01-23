set -e
#grab the lexicons from
LEXICONS_SRC="/Users/samir/Dev/resources/lexicons"
#place lexicons at
LEXICONS_DST="/Users/samir/Dev/projects/ASMAT/experiments/low_resource/input"
#grab datasets from
DATASETS_SRC="/Users/samir/Dev/projects/ASMAT/DATA/raw_datasets/"

#place datasets at
DATASETS_DST="/Users/samir/Dev/projects/ASMAT/experiments/low_resource/DATA/txt"
# DATA="/Users/samir/Dev/projects/ASMAT/experiments/asma/DATA"




#####
# Add CASM datasets
####

CLEAN=1
GET_DATA=1

if (($CLEAN > 0)); then
	rm -rf $DATASETS_DST
fi

if (($GET_DATA > 0)); then
	rm -rf $DATASETS_DST

	HCR=$DATASETS_SRC"/hcr"
	OMD=$DATASETS_SRC"/debate08"
	SEMEVAL=$DATASETS_SRC"/semeval_sentiment_raw"	
	rm -rf $DATASETS_DST	
	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-input $HCR"/dev.xml" $HCR"/test.xml" $HCR"/train.xml" \
										-outfolder $DATASETS_DST -outname "HCR.txt" 

	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/HCR.txt"  \
											 -train $DATASETS_DST"/HCR_train" \
											 -test $DATASETS_DST"/HCR_test" \
											 -dev $DATASETS_DST"/HCR_dev"		

	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-input $OMD"/dev.xml" $OMD"/test.xml" $OMD"/train.xml" \
										-outfolder $DATASETS_DST -outname "OMD.txt" 

	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/OMD.txt"  \
											 -train $DATASETS_DST"/OMD_train" \
											 -test $DATASETS_DST"/OMD_test" \
											 -dev $DATASETS_DST"/OMD_dev"		

	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-input $SEMEVAL"/Twitter2013_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW13.txt" 
	
	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/TW13.txt"  \
											 -train $DATASETS_DST"/TW13_train" \
											 -test $DATASETS_DST"/TW13_test" \
											 -dev $DATASETS_DST"/TW13_dev"				
	
	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-input $SEMEVAL"/Twitter2014_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW14.txt" 
	
	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/TW14.txt"  \
											 -train $DATASETS_DST"/TW14_train" \
											 -test $DATASETS_DST"/TW14_test" \
											 -dev $DATASETS_DST"/TW14_dev"				
	
	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-input $SEMEVAL"/Twitter2015_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW15.txt" 
	
	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/TW15.txt"  \
											 -train $DATASETS_DST"/TW15_train" \
											 -test $DATASETS_DST"/TW15_test" \
											 -dev $DATASETS_DST"/TW15_dev"				
fi
