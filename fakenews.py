# necessary imports to make
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# reads the data
df = pd.read_csv(r'C:\\Users\\pasha\\PycharmProjects\\FakeNewsDetector\\news.csv')

print(df.shape)
print(df.head())

# get the labels
labels = df.label
print(labels.head())

# this will split the data sets
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#this will initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words= 'english', max_df=0.7)

#this will fit and transform the train and test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#this will initialize a PassiveAgressiveClassifer
pac=PassiveAggressiveClassifier(max_iter=20)
pac.fit(tfidf_train,y_train)

#this will predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])




