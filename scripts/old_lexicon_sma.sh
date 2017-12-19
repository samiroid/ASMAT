mkdir -p experiments/lexicon_sma

#evaluate small lexicon
python code/lexicon_based.py -lex DATA/lexicons/semlex.txt -ts DATA/test/HCR_test.txt -res experiments/lexicon_sma/results