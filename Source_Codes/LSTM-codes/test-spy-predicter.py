from keras.layers import LSTM, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from attention import get_attention
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import numpy as np
import pickle

def convert_to_classes(test_data):
    classes = []
    for index in range(1, len(test_data)):
        if test_data[index - 1] < test_data[index]:
            classes.append(1)
        else:
            classes.append(0)

    return np.array(classes)

with open("train_data.pkl", "rb") as a:
    ac_train_data = pickle.load(a)

training_size = 1200
predict_next = 100

inputs, outputs = [], []
for index in range(6, len(ac_train_data)):
    inputs.append(ac_train_data[index-6:index, 0])
    outputs.append(ac_train_data[index, 0])
inputs, outputs = np.array(inputs), np.array(outputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))

test_X, test_Y = inputs[training_size:training_size + predict_next], \
                 outputs[training_size: training_size + predict_next]

predictor = Sequential()
predictor.add(LSTM(units = 128, return_sequences = True, input_shape = (6, 1)))
predictor.add(get_attention(activation_func='sigmoid', no_of_cells=128))
predictor.add(LSTM(units = 64, return_sequences = True))
predictor.add(Dropout(0.2))
predictor.add(LSTM(units = 32, return_sequences = True))
predictor.add(Dropout(0.2))
predictor.add(LSTM(units = 16))
predictor.add(Dense(units = 1))
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
predictor.compile(optimizer = adam, loss = 'mean_squared_error')
predictor.load_weights("weights.h5")

predictions = predictor.predict(test_X)
predictions = convert_to_classes(predictions)
print(convert_to_classes(test_Y))
print(classification_report(convert_to_classes(test_Y), predictions))
print("precision \t ", precision_score(convert_to_classes(test_Y), predictions, average='micro'))
print("recall \t ", recall_score(convert_to_classes(test_Y), predictions, average='micro'))
print("f1_score \t ", f1_score(convert_to_classes(test_Y), predictions, average='micro',
                              labels=np.unique(predictions)))
