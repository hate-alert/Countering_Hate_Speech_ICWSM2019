from .tokenize import *  
from .commen_preprocess import *
from .google_embed import *
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import *
ps = PorterStemmer()
from scipy.sparse import vstack, hstack
from os import path
import pickle 
import os

path_new=''

def get_data(pd_train):
    comments=pd_train['text'].values
    labels=pd_train['class'].values
    list_comment=[]
    for comment,label in zip(comments,labels):
        temp={}
        temp['text']=comment
        temp['label']=label
        list_comment. append(temp)
    return list_comment    

TOKENIZER = glove_tokenize
#google encoding used where text is cleaned  
def gen_data_google(pd_train):
    comments = get_data(pd_train)
    X, y = [], []
    for comment in comments:
        y.append(comment['label'])
        #X.append(tokenizer(comment['text']))
        X.append(clean(comment['text'], remove_stopwords=True, remove_punctuations=True))
    
    X =get_embeddings(X)
    return X, y


#google encoding used where text is not cleaned 
def gen_data_google2(pd_train):
    comments = get_data(pd_train)
    X, X1, y = [],[],[]
    for comment in comments:
        y.append(comment['label'])
        X.append(clean(comment['text'], remove_stopwords=False, remove_punctuations=False))
    #Word Level Features
    X =get_embeddings(X)
    return X,y

### tfidf feature generation was used here where stopwords and punctuations are removed 
def gen_data_new_tfidf(pd_train):
    comments = get_data(pd_train)
    X, y = [], []
    for comment in comments:
        y.append(comment['label'])
        X.append(comment['text'])
    
    if(path.exists(path_new+"tfidf_word_vectorizer.pk")):
            with open(path_new+'tfidf_word_vectorizer.pk', 'rb') as fin:
                word_vectorizer = pickle.load(fin)
    else:
            #Word Level Features
            word_vectorizer = TfidfVectorizer(sublinear_tf=False, ngram_range=(1,3),
                        min_df=1, 
                        strip_accents='unicode',
                        #smooth_idf=1,
                        analyzer='word', 
                        stop_words='english',
                        tokenizer=TOKENIZER,             
                        max_features=500)
            word_vectorizer.fit(X)
            with open(path_new+'tfidf_word_vectorizer.pk', 'wb') as fin:
                    pickle.dump(word_vectorizer, fin)
    
    
    
    if(path.exists(path_new+"tfidf_char_vectorizer.pk")):
            with open(path_new+'tfidf_char_vectorizer.pk', 'rb') as fin:
                char_vectorizer = pickle.load(fin)
    else:
            #Word Level Features
            char_vectorizer = TfidfVectorizer(
                            sublinear_tf=False,
                            strip_accents='unicode',
                            analyzer='char',
                            #stop_words='english',
                            ngram_range=(2, 6),
                            max_features=500)
            char_vectorizer.fit(X)
            with open(path_new+'tfidf_char_vectorizer.pk', 'wb') as fin:
                    pickle.dump(char_vectorizer, fin)
    
            
      
    
    #charlevel features new
    test_word_features = word_vectorizer.transform(X)
    test_char_features = char_vectorizer.transform(X)
    
        
    X = list(hstack([test_char_features, test_word_features]).toarray())
    return X, y

### tfidf feature generation was used here where stopwords and punctuations are not removed 
def gen_data_new_tfidf2(pd_train):
    comments = get_data(pd_train)
    X, y = [], []
    for comment in comments:
        y.append(comment['label'])
        X.append(comment['text'])
    
    if(path.exists(path_new+"tfidf_word_vectorizer.pk")):
            with open(path_new+'tfidf_word_vectorizer.pk', 'rb') as fin:
                word_vectorizer = pickle.load(fin)
    else:
            #Word Level Features
            word_vectorizer = TfidfVectorizer(sublinear_tf=False, ngram_range=(1,3),
                        min_df=1, 
                        strip_accents='unicode',
                        #smooth_idf=1,
                        analyzer='word', 
                        stop_words='english',
                        tokenizer=glove_tokenize_norem,             
                        max_features=500)
            word_vectorizer.fit(X)
            with open(path_new+'tfidf_word_vectorizer.pk', 'wb') as fin:
                    pickle.dump(word_vectorizer, fin)
    
    
    
    if(path.exists(path_new+"tfidf_char_vectorizer.pk")):
            with open(path_new+'tfidf_char_vectorizer.pk', 'rb') as fin:
                char_vectorizer = pickle.load(fin)
    else:
            #Word Level Features
            char_vectorizer = TfidfVectorizer(
                            sublinear_tf=False,
                            strip_accents='unicode',
                            analyzer='char',
                            #stop_words='english',
                            ngram_range=(2, 6),
                            max_features=500)
            char_vectorizer.fit(X)
            with open(path_new+'tfidf_char_vectorizer.pk', 'wb') as fin:
                    pickle.dump(char_vectorizer, fin)
    
            
      
    
    #charlevel features new
    test_word_features = word_vectorizer.transform(X)
    test_char_features = char_vectorizer.transform(X)
    
        
    X = list(hstack([test_char_features, test_word_features]).toarray())
    return X, y




def gen_data_embed(pd_train,word2vec_model):
    comments = get_data(pd_train)
    X, y = [], []
    for comment in comments:
        words = glove_tokenize_embed(comment['text'].lower())
        emb = np.zeros(word2vec_model['word'].shape[0])
        for word in words:
            try:
                emb += word2vec_model[word]
            except:
                pass
        if(len(words)>0):
            emb /= len(words)
            
        X.append(emb)
        y.append(comment['label'])

    return X, y


## combination of not cleaned google encodings and tfidf features where stopwords and punctuations are not removed 
def combine_tf_google_rem(pd_train):
    X,_=gen_data_google(pd_train)
    X1,y=gen_data_new_tfidf(pd_train)
    X=np.concatenate((np.array(X), np.array(X1)), axis=1)
    return X,y

## combination of cleaned google encodings and tfidf features where stopwords and punctuations are ssremoved 
def combine_tf_google_norem(pd_train):
    X,_=gen_data_google2(pd_train)
    X1,y=gen_data_new_tfidf2(pd_train)
    X=np.concatenate((np.array(X), np.array(X1)), axis=1)
    return X,y
## combination of google encodings where stopwords and punctuation are kept and tfidf features where stopwords and punctuations are removed 
def combine_tf_rem_google_norem(pd_train):
    X,_=gen_data_google2(pd_train) 
    X1,y=gen_data_new_tfidf(pd_train)
    X=np.concatenate((np.array(X), np.array(X1)), axis=1)
    return X,y
## combination of google encodings where stopwords and punctuation are removed and tfidf features where stopwords and punctuations are kept 
def combine_tf_norem_google_rem(pd_train):
    X,_=gen_data_google(pd_train)
    X1,y=gen_data_new_tfidf2(pd_train)
    X=np.concatenate((np.array(X), np.array(X1)), axis=1)
    return X,y

## combination of google encodings where stopwords and punctuation are removed and average word embeddings  
def combine_google_rem_embed(pd_train,word2vec_model):
    X,_=gen_data_google(pd_train)
    X1,y=gen_data_embed(pd_train,word2vec_model)
    X=np.concatenate((np.array(X), np.array(X1)), axis=1)
    return X,y
## combination of tfidf features where stopwords and punctuation are removed and average word embeddings  
def combine_tf_rem_embed(pd_train,word2vec_model):
    X,_=gen_data_new_tfidf(pd_train)
    X1,y=gen_data_embed(pd_train,word2vec_model)
    X=np.concatenate((np.array(X), np.array(X1)), axis=1)
    return X,y

####combination of three
def combine_tf_rem_google_norem_embed(pd_train,word2vec_model):
    X,_=gen_data_google2(pd_train)
    X1,y=gen_data_new_tfidf(pd_train)
    X2,_=gen_data_embed(pd_train,word2vec_model)
    X=np.concatenate((np.array(X), np.array(X1),np.array(X2)), axis=1)
    return X,y

def combine_tf_rem_google_rem_embed(pd_train,word2vec_model):
    X,_=gen_data_google(pd_train)
    X1,y=gen_data_new_tfidf(pd_train)
    X2,_=gen_data_embed(pd_train,word2vec_model)
    X=np.concatenate((np.array(X), np.array(X1),np.array(X2)), axis=1)
    return X,y




###old tfidf

def gen_data_old_tfidf(pd_train):
    comments = get_data(pd_train)
    X, y = [], []
    for comment in comments:
        y.append(comment['label'])
        X.append(comment['text'])
    with open('tfidf_word_vectorizer.pk', 'rb') as fin:
        word_vectorizer = pickle.load(fin)

    with open('tfidf_char_vectorizer.pk', 'rb') as fin:
        char_vectorizer = pickle.load(fin)


    
    word_vectorizer.fit(X)
    char_vectorizer.fit(X)
    
    test_word_features = word_vectorizer.transform(X)
    test_char_features = char_vectorizer.transform(X)
    X = list(hstack([test_char_features, test_word_features]).toarray())
    
    return X, y
