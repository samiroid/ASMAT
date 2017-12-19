set -e
GET_LEXICONS=0
#grab the lexicons from
LEXICONS_SRC="/Users/samir/Dev/resources/lexicons/"
#place lexicons in a local folder
LEXICONS_DST="DATA/lexicons/"

if (($GET_LEXICONS > 0)); then
	
	LEXICONS="sentiment/MPQA.txt sentiment/semlex.txt  \
			  sentiment/SentimentEmbeddingsLex.txt sentiment/OpinionMiningLex \
			  emotion/LabMT.txt emotion/ANEW emotion/NRCEmolex.txt"
	rm -rf $LEXICONS_DST
	mkdir -p $LEXICONS_DST
	# semlex is already in the correct format	
	cp $LEXICONS_SRC"sentiment/semlex.txt" $LEXICONS_DST"semlex.txt"
	#preprocess lexicons
	python ASMAT_utils/parsers/lexicon_parser.py -type mpqa -lex $LEXICONS_SRC"sentiment/MPQA.txt" -out $LEXICONS_DST		
	python ASMAT_utils/parsers/lexicon_parser.py -type oml -lex $LEXICONS_SRC"sentiment/OpinionMiningLex/" -out $LEXICONS_DST	
	python ASMAT_utils/parsers/lexicon_parser.py -type emolex -lex $LEXICONS_SRC"emotion/NRCEmolex.txt" -out $LEXICONS_DST	
	python ASMAT_utils/parsers/lexicon_parser.py -type anew -lex $LEXICONS_SRC"emotion/ANEW/anew.csv" -out $LEXICONS_DST	
	python ASMAT_utils/parsers/lexicon_parser.py -type anew_2013 -lex $LEXICONS_SRC"emotion/ANEW/anew_2013.csv" -out $LEXICONS_DST	
	python ASMAT_utils/parsers/lexicon_parser.py -type labmt -lex $LEXICONS_SRC"emotion/LabMT.txt" -out $LEXICONS_DST	
fi


GET_DATASETS=1
if (($GET_DATASETS > 0)); then
	HCR="DATA/raw_datasets/hcr/"
	OMD="DATA/raw_datasets/debate08/"
	SEMEVAL="DATA/raw_datasets/semeval_sentiment_raw/"
	STANCE="DATA/raw_datasets/semeval_stance_2016/"
	OUTFOLDER="experiments/datasets/"
	rm -rf $OUTFOLDER

	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-input $HCR"dev.xml" $HCR"test.xml" $HCR"train.xml" \
										-outfolder $OUTFOLDER -outname "HCR.txt" 

	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-input $OMD"dev.xml" $OMD"test.xml" $OMD"train.xml" \
										-outfolder $OUTFOLDER -outname "OMD.txt" 

	#convert lexicons into "datasets" for the lexicon expansion models
	for f in "$LEXICONS_DST"/*;do
		# echo $f						
		dst=$OUTFOLDER$(basename $f)
		echo ">" $dst
		python ASMAT/toolkit/lexicon_parser.py -lex $f -save_dataset $dst -out ""		
	done
fi

#grab embeddings
GET_EMBEDDINGS=0
if (($GET_EMBEDDINGS > 0)); then
	mkdir -p "DATA/embeddings"	
	EMBEDDINGS_SRC="/Users/samir/Dev/resources/embeddings/twitter/str_skip/str_skip_50.txt"
	EMBEDDINGS_DST="DATA/embeddings/str_skip_50.txt"
	echo "embeddings: "$EMBEDDINGS_SRC" -> "$EMBEDDINGS_DST
	cp $EMBEDDINGS_SRC $EMBEDDINGS_DST
fi



#####
# Add datasets
####



