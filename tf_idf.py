import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

import re
import string

import nltk
from sklearn.utils import shuffle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words('english')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

import gensim
from gensim.models import Word2Vec

#convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text
 
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))

def get_train():

    # Import the train set
    with open('new_train.json', 'r') as file:
        data = json.load(file)

    # Convert data
    f_data = {}
    for k, v in data.items():
        f_data[k] = v[:95]

    processed_data = {
        'text': [], 'target': []
    }
    for k, v in f_data.items():
        processed_data['text'] += v
        processed_data['target'] += [k] * len(v)

    df = pd.DataFrame.from_dict(processed_data)
    df['clean_text'] = df['text'].apply(lambda x: finalpreprocess(x))

    # X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['target'].values, test_size=0,
    #                                                    random_state=123, stratify=df['target'].values)

    X_train = df['clean_text'].values
    y_train = df['target'].values
    X_train, y_train = shuffle(X_train, y_train)

    return X_train, y_train

# with open('test_shuffle.txt', 'w') as file:
#     for item in X_train.tolist():
#         file.write(item + "\n")

def get_test():

    # Import the train set
    with open('test.json', 'r') as file:
        data = json.load(file)

    # Convert data
    f_data = {}
    for k, v in data.items():
        f_data[k] = v[:95]

    processed_data = {
        'text': [], 'target': []
    }
    for k, v in f_data.items():
        processed_data['text'] += v
        processed_data['target'] += [k] * len(v)

    df = pd.DataFrame.from_dict(processed_data)
    df['clean_text'] = df['text'].apply(lambda x: finalpreprocess(x))

    # X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['target'].values, test_size=0,
    #                                                    random_state=123, stratify=df['target'].values)

    X_test = df['clean_text'].values
    y_test = df['target'].values
    X_test, y_test = shuffle(X_test, y_test)

    return X_test, y_test


X_train, y_train = get_train()
X_test, y_test = get_test()

with open('y_test_shuffle.txt', 'w') as file:
    for item in y_train.tolist():
        file.write(item + "\n")

#Word2Vec
# Word2Vec runs on tokenized sentences
X_train_tok= [nltk.word_tokenize(i) for i in X_train]  
X_test_tok= [nltk.word_tokenize(i) for i in X_test]

tfidf_vectorizer = TfidfVectorizer()

tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)

tfidf_test_vectors = tfidf_vectorizer.transform(X_test)

#building Word2Vec model
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

model = Word2Vec(X_train,min_count=1)   
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors)) 
modelw = MeanEmbeddingVectorizer(w2v)# converting text to numerical data using Word2Vec
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_test_vectors_w2v = modelw.transform(X_test_tok)

classifier = RandomForestClassifier()

classifier.fit(tfidf_train_vectors, y_train)

y_pred = classifier.predict(tfidf_test_vectors)

with open('y_pred_shuffle.txt', 'w') as file:
    for item in y_pred.tolist():
        file.write(item + "\n")

print(classification_report(y_test, y_pred))


