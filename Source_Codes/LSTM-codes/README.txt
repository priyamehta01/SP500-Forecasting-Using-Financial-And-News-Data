Note:
----

simple run -> Python <filename>

1. SPY-values.csv should be in the same path as the source codes
2. format_data.py, train_spy-predictor.py and test_spy_predictor.py uses pickle files which are provided in the folder
3. Re-formatting and re-training will create new pickle files which might not give the exact same results as reported in the report, as neural networks are randomly initialized
4. For testing, just execute test-spy-predictor.py, which outputs the complete metrics report including f1-score, precion, recall, accuracy and support 

## suggestion ##
Directly execute test_spy_predictor.py as the weights of the best model is provided.


format_data.py (simple run)
--------------

1. Formats S&P 500 data and prepares it for training
2. Currently hard-coded for resolution 5, which is reported in our report
3. Reolution can be changed in line 94, x, y, z = prepare_train_test_data(5), change 5 to desired resolution


train_spy_predictor.py (simple run)
----------------------

1. Trains the stacked LSTM network with attention on the formatted data
2. The hyperparameters can be changed at the required places
3. Train function is called in line 70, train_predictor(1200, inputs, outputs, 100), currently trains using 1200 and tests on next 100
4. The weights of the best model was saved as weights.h5, retraining will change it and might not give the same results as reported in the report
5. Displays the training phase and in the end shows the loss graph with respect to epochs

test_spy_predictor.py (simple run)
---------------------

1. Tests the trained model on testing data
2. Currently tests on the next 100 instances
3. Takes the weights of best saved model (currently weights.h5) to forecast
4. Outputs the complete evaluation metrics


visualize_data.py (simple run)
----------------

1. Visualize the clossing values with resolution 1 and resolution 5

side_exps.py (simple run)
------------

1. Visualize the correct and wrong predictions with each future timestep

attention.py
------------

1. Implementation of Bahadanu additive attention