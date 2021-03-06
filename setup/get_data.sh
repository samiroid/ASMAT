set -e
#grab the lexicons from
LEXICONS_SRC="/Users/samir/Dev/resources/lexicons"
#place lexicons at
LEXICONS_DST="DATA/lexicons"
#grab datasets from

#place datasets at
DATASETS_SRC="DATA/raw_datasets"
DATASETS_DST="experiments/datasets"

NRC_HS="/Users/samir/Dev/resources/lexicons/bkp/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt"
NRC_S140="/Users/samir/Dev/resources/lexicons/bkp/Sentiment140/unigrams-pmilexicon.txt"
python ASMAT/toolkit/lexicon_parser.py -lex $NRC_HS -out $LEXICONS_DST"/NRC_HS.txt"
python ASMAT/toolkit/lexicon_parser.py -lex $NRC_S140 -out $LEXICONS_DST"/NRC_S140.txt"

GET_LEXICONS=0
if (($GET_LEXICONS > 0)); then	
	LEXICONS="sentiment/MPQA.txt sentiment/semlex.txt  \
			  sentiment/SentimentEmbeddingsLex.txt sentiment/OpinionMiningLex \
			  emotion/LabMT.txt emotion/ANEW emotion/NRCEmolex.txt"
	rm -rf $LEXICONS_DST
	mkdir -p $LEXICONS_DST
	# semlex is already in the correct format	
	cp $LEXICONS_SRC"sentiment/semlex.txt" $LEXICONS_DST"/semlex.txt"
	#preprocess lexicons
	python ASMAT_utils/parsers/lexicon_parser.py -type mpqa -lex $LEXICONS_SRC"/sentiment/MPQA.txt" -out $LEXICONS_DST		
	python ASMAT_utils/parsers/lexicon_parser.py -type oml -lex $LEXICONS_SRC"sentiment/OpinionMiningLex/" -out $LEXICONS_DST	
	python ASMAT_utils/parsers/lexicon_parser.py -type emolex -lex $LEXICONS_SRC"emotion/NRCEmolex.txt" -out $LEXICONS_DST	
	python ASMAT_utils/parsers/lexicon_parser.py -type anew -lex $LEXICONS_SRC"/emotion/ANEW/anew.csv" -out $LEXICONS_DST	
	python ASMAT_utils/parsers/lexicon_parser.py -type anew_2013 -lex $LEXICONS_SRC"emotion/ANEW/anew_2013.csv" -out $LEXICONS_DST	
	python ASMAT_utils/parsers/lexicon_parser.py -type labmt -lex $LEXICONS_SRC"/emotion/LabMT.txt" -out $LEXICONS_DST	
fi

#####
# Add datasets
####

GET_DATASETS=0
if (($GET_DATASETS > 0)); then
	HCR=$DATASETS_SRC"/hcr"
	OMD=$DATASETS_SRC"/debate08"
	SEMEVAL=$DATASETS_SRC"/semeval_sentiment_raw"
	STANCE=$DATASETS_SRC"/semeval_stance_2016"	
	rm -rf $DATASETS_DST
	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-input $HCR"/dev.xml" $HCR"/test.xml" $HCR"/train.xml" \
										-outfolder $DATASETS_DST -outname "HCR.txt" 

	python ASMAT/toolkit/dataset_parser.py -format "omd_hcr" \
										-input $OMD"/dev.xml" $OMD"/test.xml" $OMD"/train.xml" \
										-outfolder $DATASETS_DST -outname "OMD.txt" 

	#convert lexicons into "datasets" for the lexicon expansion models
	for f in "$LEXICONS_DST"/*;do
		# echo $f						
		dst=$DATASETS_DST"/"$(basename $f)
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