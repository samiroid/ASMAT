from collections import Counter, defaultdict

def build_vocabulary(docs, zero_for_padd=False, max_words=None):
    """
        Compute a dictionary index mapping words into indices
    """
    words = [w for m in docs for w in m.split()]
    #keep only the 'max_words' most frequent tokens
    if max_words is not None:
        top_words = sorted(Counter(words).items(), key=lambda x:x[1],reverse=True)[:max_words]
        words = [w[0] for w in top_words]
    #keep only the types
    words = list(set(words))
    #prepend the padding token
    if zero_for_padd: words = ['_PAD_'] + words
    vocab = {w:i for i, w in enumerate(words)}
    return vocab

def idx2word(wrd2idx):
    '''

    '''

    return {i:w for w, i in wrd2idx.items()}


def docs2idx(docs, wrd2idx=None):
    """
        Convert documents to lists of word indices
    """
    if wrd2idx is None:
        wrd2idx = build_vocabulary(docs)
    
    X = [[wrd2idx[w] for w in m.split() if w in wrd2idx] for m in docs]
    return X, wrd2idx


# def docs2idx(docs, labels, wrd2idx=None, lbl2idx=None):
#     """
#         Convert documents and labels to indices
#     """
#     if wrd2idx is None: wrd2idx = word2idx(docs)
#     if lbl2idx is None: lbl2idx = word2idx(labels)
#     X = [[wrd2idx[w] for w in m.split() if w in wrd2idx] for m in docs]
#     Y = [lbl2idx[l] for l in labels]

#     return X, Y, wrd2idx, lbl2idx
