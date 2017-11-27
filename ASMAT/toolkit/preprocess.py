import re
from ..ext.tweetokenize import Tokenizer
from ..ext import twokenize

# emoticon regex taken from Christopher Potts' script at http://sentiment.christopherpotts.net/tokenizing.html
emoticon_regex = r"""(?:[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?)"""

twk = Tokenizer(ignorequotes=False,usernames=False,urls=False,numbers=False)


def max_reps(sentence, n=3):

    """
        Normalizes a string to at most n repetitions of the same character
        e.g, for n=3 and "helllloooooo" -> "helllooo"
    """
    new_sentence = ''
    last_c = ''
    max_counter = n
    for c in sentence:
        if c != last_c:
            new_sentence+=c
            last_c = c
            max_counter = n
        else:
            if max_counter > 1:
                new_sentence+=c
                max_counter-=1
            else:
                pass
    return new_sentence

def preprocess(m, sep_emoji=False):
    assert type(m) == unicode
    
    m = m.lower()    
    m = max_reps(m)
    #replace user mentions with token '@user'
    user_regex = r".?@.+?( |$)|<@mention>"    
    m = re.sub(user_regex," @user ", m, flags=re.I)
    #replace urls with token 'url'
    m = re.sub(twokenize.url," url ", m, flags=re.I)        
    tokenized_msg = ' '.join(twokenize.tokenize(m)).strip()
    if sep_emoji:
        #tokenize emoji, this tokenzier however has a problem where repeated punctuation gets separated e.g. "blah blah!!!"" -> ['blah','blah','!!!'], instead of ['blah','blah','!','!','!']
        m_toks = tokenized_msg.split()
        n_toks = twk.tokenize(tokenized_msg)         
        if len(n_toks)!=len(m_toks):
            #check if there is any punctuation in this string
            has_punct = map(lambda x:x in twk.punctuation, n_toks)
            if any(has_punct):  
                new_m = n_toks[0]
                for i in xrange(1,len(n_toks)):
                    #while the same punctuation token shows up, concatenate
                    if has_punct[i] and has_punct[i-1] and (n_toks[i] == n_toks[i-1]):
                        new_m += n_toks[i]
                    else:
                        #otherwise add space
                        new_m += " "+n_toks[i]                   
                tokenized_msg = new_m                
    return tokenized_msg.lstrip()