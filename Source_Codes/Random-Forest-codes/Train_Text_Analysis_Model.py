import re
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import f1_score

stemmer = WordNetLemmatizer()

path = "D:/Academic/UT Dallas Curriculum/Semester 1/Machine Learning/Project/Dataset/Stock Market DJIA News Sentiment Dataset/combined_stock_data.csv"
df = pd.read_csv(path, encoding="utf-8")


def preprocess(line):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(line))

    # remove all single characters
    document = re.sub('\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    # filtered_words = [word for word in document if word not in stopwords.words('english')]
    document = ' '.join(document)
    return document


dic = {}
dic_labels = {}
for idx, row in df.iterrows():
    date = row["Date"]
    label = row["Label"]
    dic[date] = [preprocess(row["Top1"])]
    for i in range(2, 26):
        st = "Top" + str(i)
        dic[date].append(preprocess(row[st]))

    dic_labels[date] = label

df = pdr.get_data_yahoo(symbols='SPY', start=datetime(2008, 8, 8), end=datetime(2016, 7, 1))

df['Date'] = pd.to_datetime(df.index) # condf.indexvert col Date to datetime
df.set_index("Date",inplace=True) # set col Date as index
df = df.reset_index()

for date in dic:
    rowIndex = df.index[df.Date == pd.Timestamp(date)]
    df.loc[rowIndex, 'News'] = " ".join(i for i in dic[date])
    df.loc[rowIndex, 'Label'] = dic_labels[date]

df.loc[0, 'Up_Down'] = 0

for index, row in df.iterrows():
    if index != 0:
        curr_close = df.loc[index, "Adj Close"]
        prev_close = df.loc[index - 1, "Adj Close"]
        if curr_close > prev_close:
            up_down = 1

        else:
            up_down = 0

        df.loc[index, 'Up_Down'] = up_down

data = []

for idx, row in df.iterrows():
    label = row["Up_Down"]
    news = row["News"]
    data.append([news, label])

news_data = [x[0] for x in data]
label_data = [x[1] for x in data]

X_train, X_test, y_train, y_test = train_test_split(news_data, label_data, train_size = 0.75)

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
vectorizer.fit(X_train)

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train_tfidf, y_train)

score = accuracy_score(y_test, classifier.predict(X_test_tfidf))
print("Score  : ", score)
f1_score_model = f1_score(y_test, classifier.predict(X_test_tfidf))
print("F1 Score : ", f1_score_model)

# save_classifier = open("C:/Users/asus/PycharmProjects/test/text_classification_random_forest.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()   
#
# with open('C:/Users/asus/PycharmProjects/test/tf_idf_vectorizer.pk', 'wb') as fin:
#     pickle.dump(vectorizer, fin)