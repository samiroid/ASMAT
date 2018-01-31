import codecs
import sys
sys.path.append("/Users/samir/Dev/projects/ASMAT/")
from ASMAT.lib.preprocess import preprocess

LEX="/Users/samir/Dev/projects/ASMAT/DATA/lexicons/NRC_S140.txt"
LEX_2="/Users/samir/Dev/projects/ASMAT/DATA/lexicons/NRC_S140_2.txt"
with codecs.open(LEX,"r","utf-8") as fid:
    with codecs.open(LEX_2,"w","utf-8") as fod:
        for l in fid:     
            wd, sc = l.split("\t")
            n_wd = preprocess(wd)
            if n_wd in ["@user","url"]:                
                continue
            fod.write(u"{}\t{}".format(wd,sc))
            # print wd, " -- ", n_wd