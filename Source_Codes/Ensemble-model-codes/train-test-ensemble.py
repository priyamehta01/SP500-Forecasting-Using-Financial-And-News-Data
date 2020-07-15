import pandas as pd
import numpy as np

df = pd.read_excel("ensem_data.xlsx")

Y = df["actual values"][:-3]
X = df[['nn predicted float', 'random forest predicted values']][:-3]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1000))
scaler = scaler.fit(df[['nn predicted float']])
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
normalized = scaler.transform(df[['nn predicted float']])
df[['nn predicted float']] = normalized

Y = df["actual values"][:-3]
X = df[['nn predicted float', 'random forest predicted values']][:-3]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)

from sklearn.svm import SVC
svclassifier = SVC()
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("precision \t ", precision_score(y_test, y_pred, average='micro'))
print("recall \t ", recall_score(y_test, y_pred, average='micro'))
print("f1_score \t ", f1_score(y_test, y_pred, average='micro',
                              labels=np.unique(y_pred)))
