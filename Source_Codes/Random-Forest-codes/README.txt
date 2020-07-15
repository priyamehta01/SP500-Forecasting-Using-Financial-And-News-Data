Train_Text_Analysis_Model.py 

This file contains the code for training the text analysis model. A
random forest classifier is trained on the TFIDF values of the news 
headlines from the DJIA stock market sentiment data. The labels are
generated from the Yahoo Finance API, the labels being the stock 
market going up or down from the previous day. The model calculates
precision, recall and F-1 score values. Model and TFIDF vectorizer 
weights are pickled. The file requires you to input the path to the 
DJIA stock dataset and the path where the pickle files need to be 
stored. 

------

News_Prediction_Statistics.py

This file loads the model and the TFIDF vectorizer weights file and 
generates predictions on the Reuters News headlines dataset. Various
preprocessing techniques are employed to clean the data. The model is
evaluated on precision, recall and F-1 Score. The model requires you 
to give the path to the pickle files and the Reuters News Headlines 
dataset. 

------

Get_News_Prediction.py

This file scraps the headlines for the current date from news sources
related to SP500. The pickle files for the model and the vectorizer are
loaded and prediction for the market going up or down for that day is
made.
