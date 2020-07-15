ensemb_data.xlsx

Excel file containing training and testing data and the raw and classification outputs of both LSTM and Random Forest


train-test-ensemble.py (simple run)
-------------------

1. Takes the raw output from LSTM and normalizes it, and the classification output from Random Forest
2. Trains an SVM clasisifer, and outputs the evaluation metrics, f1-score, precision, recall, accuracy