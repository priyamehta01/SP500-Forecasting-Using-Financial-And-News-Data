import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras import optimizers
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from attention import get_attention
import numpy as np


with open("train_data.pkl", "rb") as a:
    ac_train_data = pickle.load(a)

def convert_to_classes(test_data):
    classes = []
    for index in range(1, len(test_data)):
        if test_data[index - 1] < test_data[index]:
            classes.append(1)
        else:
            classes.append(0)

    return np.array(classes)

inputs, outputs = [], []
for index in range(6, len(ac_train_data)):
    inputs.append(ac_train_data[index-6:index, 0])
    outputs.append(ac_train_data[index, 0])
inputs, outputs = np.array(inputs), np.array(outputs)

inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))

def train_predictor(training_size, inputs, outputs, predict_next):

    slider = 0
    train_X, train_Y = inputs[slider:slider+training_size], outputs[slider:slider+training_size]
    test_X, test_Y = inputs[training_size:training_size + predict_next], \
                     outputs[training_size: training_size + predict_next]

    predictor = Sequential()
    predictor.add(LSTM(units = 128, return_sequences = True, input_shape = (train_X.shape[1], 1)))
    predictor.add(get_attention(activation_func='sigmoid', no_of_cells=128))

    predictor.add(LSTM(units = 64, return_sequences = True))
    predictor.add(Dropout(0.2))

    predictor.add(LSTM(units = 32, return_sequences = True))
    predictor.add(Dropout(0.2))

    predictor.add(LSTM(units = 16))
    predictor.add(Dense(units = 1))

    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    predictor.compile(optimizer = adam, loss = 'mean_squared_error')

    history = predictor.fit(train_X, train_Y, epochs=25, batch_size=32, verbose=2, shuffle=False)
    print(history)
    predictor.save('model.h5')
    predictor.save_weights("weights.h5")
    print(predictor.summary())

    # LOSS
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


train_predictor(1200, inputs, outputs, 100)
