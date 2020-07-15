import pickle
import matplotlib.pyplot as plt

def convert_to_classes(test_data):
    classes = []
    for index in range(1, len(test_data)):
        if test_data[index - 1] < test_data[index]:
            classes.append(1)
        else:
            classes.append(0)
    return classes

with open("close_values.pkl", "rb") as e:
    close_vals = pickle.load(e)
with open("train_data.pkl", "rb") as a:
    train_data = pickle.load(a)

x_1 = [index for index in range(len(close_vals))]
x_2 = [index for index in range(len(train_data))]

y_1 = close_vals
y_2= [value[3] for value in train_data]

fig, axs = plt.subplots(nrows=2, ncols=1)
axs[0].plot(x_1, y_1, color='g')
axs[0].set_title("close values (daily)")

axs[1].plot(x_2, y_2, color='b')
axs[1].set_title("close values (weekly)")

plt.show()
