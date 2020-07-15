import requests
import json
import pandas
import pickle
import re
from nltk.stem import WordNetLemmatizer

headers = {'Authorization': '4d7d8134b4134170bfaec5ee56fc3443'}

top_headlines_url = 'https://newsapi.org/v2/top-headlines'
everything_news_url = 'https://newsapi.org/v2/everything'
sources_url = 'https://newsapi.org/v2/sources'

headlines_payload = {'category': 'business', 'country': 'us'}
everything_payload = {'q': 'SP500', 'language': 'en', 'sortBy': 'popularity'}
sources_payload = {'category': 'general', 'language': 'en', 'country': 'us'}

response = requests.get(url=everything_news_url, headers=headers, params=everything_payload)
pretty_json_output = json.dumps(response.json(), indent=4)

response_json_string = json.dumps(response.json())
response_dict = json.loads(response_json_string)

classifier_f = open("C:/Users/asus/PycharmProjects/test/text_classification_random_forest.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

vectorizer_f = open("C:/Users/asus/PycharmProjects/test/tf_idf_vectorizer.pk", "rb")
vectorizer = pickle.load(vectorizer_f)
vectorizer_f.close()

stemmer = WordNetLemmatizer()

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

    # document = [stemmer.lemmatize(word) for word in document]
    # filtered_words = [word for word in document if word not in stopwords.words('english')]
    document = ' '.join(document)
    return document

headlines = []
for headline in response_dict["articles"]:
    headlines.append(preprocess(headline["title"]))

headlines_data = " ".join(headlines)

classifier_f = open("C:/Users/asus/PycharmProjects/test/text_classification_random_forest.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

vectorizer_f = open("C:/Users/asus/PycharmProjects/test/tf_idf_vectorizer.pk", "rb")
vectorizer = pickle.load(vectorizer_f)
vectorizer_f.close()

news = [headlines_data]
X_test_tfidf = vectorizer.transform(news)
pred_prob = classifier.predict_proba(X_test_tfidf)
pred = classifier.predict(X_test_tfidf)

print(pred_prob)
print(pred)