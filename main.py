from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from pathlib import Path
import nltk
import re
import numpy as np
from sklearn.feature_extraction import text

import os

""" 
How to compose python pipeline-s for ML in NLTK using pipeline and FeatureUnion to add features to the processing
Approach is explained here: https://chrisfotache.medium.com/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0

Text example is loaded from here:

"""


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.field]


class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.field]]


class ProcessData(object):

    def __init__(self):
        self.complete_workflow()

    def complete_workflow(self):
        y_test, preds = self.main()
        print ("y_test=\n{} \n\npreds=\n{}".format(y_test, preds))
        self.report_results(y_test, preds)


    def report_results(self, y_test, preds):
        from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
        print ("Accuracy:", accuracy_score(y_test, preds))
        print ("Precision:", precision_score(y_test, preds))
        print
        classification_report(y_test, preds)
        print
        confusion_matrix(y_test, preds)

    def Tokenizer(self, str_input):
        words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
        porter_stemmer = nltk.PorterStemmer()
        words = [porter_stemmer.stem(word) for word in words]
        return words

    def load_data(self, input_dir):
        import glob, os
        X = pd.concat(map(lambda f: pd.read_csv(f, names=['Text'], sep='`', header=None),
                          glob.glob(os.path.join(input_dir, "*.txt"))))
        X['TotalWords'] = X.apply(lambda row: len(row['Text'].split()), axis=1)
        return X

    def ingest_data(self):
        input_dir_pos_name = "/Users/mboyanna/idea/datascience-light/data/aclImdb/train/pos"
        input_dir_neg_name = "/Users/mboyanna/idea/datascience-light/data/aclImdb/train/neg"

        X_arr = [self.load_data(input_dir_pos_name), self.load_data(input_dir_neg_name)]
        num_rows_pos, num_rows_neg = X_arr[0].shape[0], X_arr[1].shape[0]
        X = pd.concat(X_arr)
        Y_arr = [np.ones(num_rows_pos), np.zeros(num_rows_neg)]
        Y = np.concatenate(Y_arr)
        print("Loaded {} rows".format(num_rows_pos+num_rows_neg))
        print("X_pos={}\nX_neg={}\nY_pos={}\nY_neg={}".format(X_arr[0], X_arr[1], Y_arr[0], Y_arr[1]))

        return X, Y

    def define_stop_words(self):
        my_stop_words = []
        stop_words = text.ENGLISH_STOP_WORDS.union(my_stop_words)
        return stop_words

    def main(self):

        X, Y = self.ingest_data()

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

        stop_words = self.define_stop_words()

        classifier = Pipeline([
            ('features', FeatureUnion([
                ('text', Pipeline([
                    ('colext', TextSelector('Text')),
                    ('tfidf', TfidfVectorizer(tokenizer=self.Tokenizer, stop_words=stop_words,
                                              min_df=.0025, max_df=0.25, ngram_range=(1, 3))),
                    ('svd', TruncatedSVD(algorithm='randomized', n_components=300)),  # for XGB
                ])),
                ('words', Pipeline([
                    ('wordext', NumberSelector('TotalWords')),
                    ('wscaler', StandardScaler()),
                ])),
            ])),
            ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),
            #    ('clf', RandomForestClassifier()),
        ])

        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)

        return y_test, preds

######
# Main
######

p = ProcessData()
