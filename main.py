#!/usr/bin/env python
# encoding: utf-8

"""
Script: main.py
Auteur: PF
Date: 16/04/2021 11:50
"""

# Imports
from get_data import *
from stopwords import *
import unidecode
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


# Functions

def punctuation(words):
    words_no_punc = []

    for w in words:
        if w.isalpha():
            words_no_punc.append(w)

    return words_no_punc


def accents(words):
    words_no_acc = []

    for w in words:
        if w.isalpha():
            words_no_acc.append(unidecode.unidecode(w))

    return words_no_acc


def stopw(words, lang='french'):
    if lang is 'french':
        stop_words = custom_fr_stopwords()
        words = [w for w in words if w not in stop_words]
    else:
        words = [w for w in words if w not in stopwords.words(lang)]

    return words


def stemming(words, lang='french'):
    words_stemmed = []
    snow = SnowballStemmer(lang)

    if lang is 'english':
        lemma = WordNetLemmatizer()
        for w in words:
            words_stemmed.append(lemma.lemmatize(w))
    else:
        for w in words:
            words_stemmed.append(snow.stem(w))

    return words_stemmed


def preproc_stop(text, lang='french'):
    return stemming(accents(stopw(punctuation(word_tokenize(text.lower())))), lang)


def preproc(text, lang='french'):
    return stemming(accents(punctuation(word_tokenize(text.lower()))), lang)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


# Main

def main():
    # df = get_data("test.csv")

    df = get_bbc_data()

    df.New = df.New.apply(lambda x: preproc_stop(x, lang='english'))
    df.Summary = df.Summary.apply(lambda x: preproc(x, lang='english'))

    tfidf = TfidfVectorizer(tokenizer=preproc_stop)
    tfidf.fit(df.New)

    df2 = df.copy()
    documents = df2.New.tolist()
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)

    n_topics = 5
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=5,
        learning_method='online',
        learning_offset=50.,
        random_state=0)

    lda.fit(tf)

    no_top_words = 10
    display_topics(lda, tf_vectorizer.get_feature_names(), no_top_words)

if __name__ == '__main__':
    main()
