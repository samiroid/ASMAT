set -e
# Reset
COLOR_OFF='\033[0m'       # Text Reset
# Regular Colors
RED='\033[0;31m'          # RED

#grab datasets from
DATASETS_SRC="/Users/samir/Dev/projects/ASMAT/RAW_DATA/raw_datasets/"
DATASETS_SRC="/data/ASMAT/ASMAT/RAW_DATA/raw_datasets/"
#place datasets at
DATASETS_DST="/Users/samir/Dev/projects/ASMAT/experiments/document_models/DATA/txt"
DATASETS_DST="/data/ASMAT/ASMAT/experiments/document_models/DATA/txt"

CLEAN=1
GET_DATA=1


if (($CLEAN > 0)); then
	rm -rf $DATASETS_DST
fi

if (($GET_DATA > 0)); then

	HCR=$DATASETS_SRC"/hcr"
	OMD=$DATASETS_SRC"/debate08"
	SEMEVAL=$DATASETS_SRC"/semeval_sentiment_raw"		

	echo $RED" >> HCR "$COLOR_OFF
	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-input $HCR"/dev.xml" $HCR"/test.xml" $HCR"/train.xml" \
										-outfolder $DATASETS_DST -outname "HCR.txt" 

	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/HCR.txt"  \
											 -train $DATASETS_DST"/HCR_train" \
											 -test $DATASETS_DST"/HCR_test" \
											 -dev $DATASETS_DST"/HCR_dev"		
	
	echo $RED" >> OMD "$COLOR_OFF	
	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-input $OMD"/dev.xml" $OMD"/test.xml" $OMD"/train.xml" \
										-outfolder $DATASETS_DST -outname "OMD.txt" 

	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/OMD.txt"  \
											 -train $DATASETS_DST"/OMD_train" \
											 -test $DATASETS_DST"/OMD_test" \
											 -dev $DATASETS_DST"/OMD_dev"		

	echo $RED" >> TW13 "$COLOR_OFF	
	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-input $SEMEVAL"/Twitter2013_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW13.txt" 
	
	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/TW13.txt"  \
											 -train $DATASETS_DST"/TW13_train" \
											 -test $DATASETS_DST"/TW13_test" \
											 -dev $DATASETS_DST"/TW13_dev"				
	echo $RED" >> TW14 "$COLOR_OFF	
	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-input $SEMEVAL"/Twitter2014_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW14.txt" 
	
	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/TW14.txt"  \
											 -train $DATASETS_DST"/TW14_train" \
											 -test $DATASETS_DST"/TW14_test" \
											 -dev $DATASETS_DST"/TW14_dev"				
	echo $RED" >> TW15 "$COLOR_OFF	
	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-input $SEMEVAL"/Twitter2015_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW15.txt" 
	
	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/TW15.txt"  \
											 -train $DATASETS_DST"/TW15_train" \
											 -test $DATASETS_DST"/TW15_test" \
											 -dev $DATASETS_DST"/TW15_dev"				
fi

