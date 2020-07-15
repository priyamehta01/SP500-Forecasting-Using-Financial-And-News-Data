import pickle
from nltk.stem import WordNetLemmatizer
import os
import re
import pandas_datareader as pdr
from datetime import datetime
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import numpy as np

stemmer = WordNetLemmatizer()

classifier_f = open("C:/Users/asus/PycharmProjects/test/text_classification_random_forest.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

vectorizer_f = open("C:/Users/asus/PycharmProjects/test/tf_idf_vectorizer.pk", "rb")
vectorizer = pickle.load(vectorizer_f)
vectorizer_f.close()


def preprocess(line):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(line))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', line)

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

def preprocess(line):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(line))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', line)

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

for folder in os.listdir("C:/Users/asus/Downloads/financial-news-dataset-master/ReutersNews106521/"):
    if folder != ".DS_Store":
        for file in os.listdir("C:/Users/asus/Downloads/financial-news-dataset-master/ReutersNews106521/" + folder + "/"):
            with open("C:/Users/asus/Downloads/financial-news-dataset-master/ReutersNews106521/" + folder + "/" + file) as f:
                lines = f.readlines()
                #print(lines[2])
                #print("C:/Users/asus/Downloads/financial-news-dataset-master/ReutersNews106521/" + folder + "/" + file)
                if len(lines) >= 3:
                    headline = preprocess(lines[0].split("--")[1])
                    date = " ".join(lines[2].replace(",", "").split(" ")[2:5])
                    st_date = datetime.strptime(date, '%b %d %Y')
                    #st_date = datetime.strptime(date, '%b %d %Y')
                    date = st_date.date()
                    if date not in dic:
                        dic[date] = [headline]
                    else:
                        dic[date].append(headline)

df = pdr.get_data_yahoo(symbols='SPY', start=datetime(2006, 10, 20), end=datetime(2013, 11, 19))
cols = [df.columns]

df['Date'] = pd.to_datetime(df.index) # condf.indexvert col Date to datetime
df.set_index("Date",inplace=True) # set col Date as index
df = df.resample("D").ffill().reset_index() # resample Days and fill values


for date in dic:
    rowIndex = df.index[df.Date == pd.Timestamp(date)]
    df.loc[rowIndex, 'News'] = " ".join(i for i in dic[date])

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


df = df.dropna()

data = []

for idx, row in df.iterrows():
    label = row["Up_Down"]
    news = row["News"]
    news = preprocess(news)
    data.append([news, label])

news_data = [x[0] for x in data]
label_data = [x[1] for x in data]

count = 0
pred_arr = []
for i, news in enumerate(news_data):
    news = [news]
    X_test_tfidf = vectorizer.transform(news)
    pred_prob = classifier.predict_proba(X_test_tfidf)
    pred = classifier.predict(X_test_tfidf)
    if label_data[i] == pred:
        count += 1

    pred_arr.append(pred)


print(classification_report(label_data, pred_arr))
print("precision \t ", precision_score(label_data, pred_arr, average='micro'))
print("recall \t ", recall_score(label_data, pred_arr, average='micro'))
print("f1_score \t ", f1_score(label_data, pred_arr, average='micro',
                              labels=np.unique(pred_arr)))