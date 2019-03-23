from .commen_preprocess import *
from string import punctuation
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import *
ps = PorterStemmer()
from scipy.sparse import vstack, hstack


### stopwords and punctuations are not removed but text is cleaned and stemmed
def glove_tokenize_norem(text):
    #text = tokenizer(text)
    text=clean(text, remove_stopwords=False, remove_punctuations=False)
    words = text.split()
    words =[ps.stem(word) for word in words]
    return words

####stopwords and punctuations are removed along with that text is cleaned ans stemmed
def glove_tokenize(text):
    #text = tokenizer(text)
    text=clean(text, remove_stopwords=False, remove_punctuations=False)
    text = ''.join([c for c in text if c not in punctuation])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    words =[ps.stem(word) for word in words]
    return words

### this is the glove tokenize for embedding
def glove_tokenize_embed(text):
    #text = tokenizer(text)
    text=clean(text, remove_stopwords=False, remove_punctuations=False)
    text = ''.join([c for c in text if c not in punctuation])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return words

