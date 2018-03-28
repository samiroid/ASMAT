set -e
# Reset
COLOR_OFF='\033[0m'       # Text Reset
# Regular Colors
RED='\033[0;31m'          # RED

#grab datasets from
DATASETS_SRC="/Users/samir/Dev/projects/ASMAT/DATA/raw_datasets/"
#place datasets at
DATASETS_DST="/Users/samir/Dev/projects/ASMAT/experiments/lexicon_models/DATA/txt"

CLEAN=1
GET_BINARY=1
GET_SMALLER_BINARY=0

if (($CLEAN > 0)); then
	rm -rf $DATASETS_DST
fi

#datasets for the lexicon experiments only consider binary labels 
#note thow the data_splitter is being invoked: 
#  split the data into 80% FOR TESTING and 20% FOR TUNING
if (($GET_BINARY > 0)); then

	HCR=$DATASETS_SRC"/hcr"
	OMD=$DATASETS_SRC"/debate08"
	SEMEVAL=$DATASETS_SRC"/semeval_sentiment_raw"	
	
	echo $RED" >> HCR~BINARY "$COLOR_OFF	
	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-labels positive negative \
										-input $HCR"/dev.xml" $HCR"/test.xml" $HCR"/train.xml" \
										-outfolder $DATASETS_DST -outname "HCR~BINARY.txt" 

	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/HCR~BINARY.txt"  \
											 -train $DATASETS_DST"/HCR~BINARY_test"	\
											 -test $DATASETS_DST"/HCR~BINARY_train"											 
											 										 	
	echo $RED" >> OMD~BINARY "$COLOR_OFF	
	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-labels positive negative \
										-input $OMD"/dev.xml" $OMD"/test.xml" $OMD"/train.xml" \
										-outfolder $DATASETS_DST -outname "OMD~BINARY.txt" 

	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/OMD~BINARY.txt"  \
											 -train $DATASETS_DST"/OMD~BINARY_test" \
											 -test $DATASETS_DST"/OMD~BINARY_train"

	echo $RED" >> TW13~BINARY "$COLOR_OFF	
	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-labels positive negative \
										-input $SEMEVAL"/Twitter2013_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW13~BINARY.txt" 
	
	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/TW13~BINARY.txt"  \
											 -train $DATASETS_DST"/TW13~BINARY_test" \
											 -test $DATASETS_DST"/TW13~BINARY_train" 
											 									 		
	echo $RED" >> TW14~BINARY "$COLOR_OFF	
	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-labels positive negative \
										-input $SEMEVAL"/Twitter2014_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW14~BINARY.txt" 
	
	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/TW14~BINARY.txt"  \
											 -train $DATASETS_DST"/TW14~BINARY_test" \
											 -test $DATASETS_DST"/TW14~BINARY_train"
											 
											 			
	echo $RED" >> TW15~BINARY "$COLOR_OFF	
	python ASMAT/toolkit/dataset_parser.py -format "semeval" \
										-labels positive negative \
										-input $SEMEVAL"/Twitter2015_raw.txt" \
										-outfolder $DATASETS_DST -outname "TW15~BINARY.txt" 
	
	python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/TW15~BINARY.txt"  \
											 -train $DATASETS_DST"/TW15~BINARY_test" \
											 -test $DATASETS_DST"/TW15~BINARY_train"
											 											 	
fi

#get another sample with 10% of the data to fit
if (($GET_SMALLER_BINARY > 0)); then
echo $RED" >> SMALLER DATASETS "$COLOR_OFF	
datasets="HCR OMD TW13 TW14 TW15"
	for ds in $datasets; 		
	do
		# 10% is half of 20%
		python ASMAT/toolkit/dataset_splitter.py -input $DATASETS_DST"/"$ds"~BINARY_train"  \
										  -train $DATASETS_DST"/"$ds"~BINARY_small_train" \
										  -test /dev/null \
										  -split 0.5
		cp $DATASETS_DST"/"$ds"~BINARY_test" $DATASETS_DST"/"$ds"~BINARY_small_test"

	done
fi


