import pickle
from pandas import read_csv
from pandas import DataFrame
import numpy as np


def parse_csv(path):
    with open(path) as csvfile:
        read_CSV = read_csv(csvfile, header=0, index_col=0)

    read_CSV = DataFrame(read_CSV)

    date, open_value, high_value, \
    low_value, close_value, volume = [], [], [], [], [], []

    for value in read_CSV.index.values:
        date.append(value[:-6])

    open_norm = max(list(read_CSV["Open"].values))
    for value in list(read_CSV["Open"].values):
        open_value.append(value/open_norm)

    high_norm = max(list(read_CSV["High"].values))
    for value in list(read_CSV["High"].values):
        high_value.append(value/high_norm)

    low_norm = max(list(read_CSV["Low"].values))
    for value in list(read_CSV["Low"].values):
        low_value.append(value/low_norm)

    close_norn = max(list(read_CSV["Close"].values))
    for value in list(read_CSV["Close"].values):
        close_value.append(value/close_norn)

    vol_norm = max(list(read_CSV["Volume"].values))
    for value in list(read_CSV["Volume"].values):
        volume.append(value/vol_norm)

    with open("date.pkl", "wb") as a:
        pickle.dump(date, a)
    with open("open_values.pkl", "wb") as b:
        pickle.dump(open_value, b)
    with open("high_values.pkl", "wb") as c:
        pickle.dump(high_value, c)
    with open("low_values.pkl", "wb") as d:
        pickle.dump(low_value, d)
    with open("close_values.pkl", "wb") as e:
        pickle.dump(close_value, e)
    with open("volume_values.pkl", "wb") as f:
        pickle.dump(volume, f)

def get_formatted_data(resolutions):

    with open("date.pkl", "rb") as a:
        date = pickle.load(a)
    with open("open_values.pkl", "rb") as b:
        open_values = pickle.load(b)
    with open("high_values.pkl", "rb") as c:
        high_values = pickle.load(c)
    with open("low_values.pkl", "rb") as d:
        low_values = pickle.load(d)
    with open("close_values.pkl", "rb") as e:
        close_values = pickle.load(e)

    res_open, res_high, res_low, res_close, res_vol = [], [], [], [], []
    date_ranges = []

    for index in range(0, len(open_values) - resolutions, resolutions):
        res_open.append([open_values[index]])
        res_close.append([close_values[index + resolutions]])
        dummy_high, dummy_low = 0, 9
        for dummy_index in range(resolutions):
            if high_values[int(dummy_index + index)] > dummy_high:
                dummy_high = high_values[dummy_index + index]
            if low_values[int(dummy_low + index)] < dummy_low:
                dummy_low = low_values[dummy_index + index]
        res_high.append([dummy_high])
        res_low.append([dummy_low])
        date_ranges.append([date[index], date[index + resolutions]])

    return date_ranges, np.array(res_open), np.array(res_high), \
           np.array(res_low), np.array(res_close)

def prepare_train_test_data(resolutions):
    date_ranges, res_open, res_high, res_low, res_close = get_formatted_data(resolutions)
    res_t_open, res_t_high, res_t_low, res_t_close = res_open[1:], res_high[1:], \
                                                     res_low[1:], res_close[1:]

    return np.concatenate((res_open, res_high, res_low, res_close), axis=1)\
        , np.concatenate((res_t_open, res_t_high, res_t_low, res_t_close), axis=1)\
        , date_ranges


x, y, z = prepare_train_test_data(5)


with open("train_data.pkl", "wb") as a:
    pickle.dump(x, a)

with open("test_data.pkl", "wb") as b:
    pickle.dump(y, b)

with open("date_ranges.pkl", "wb") as c:
    pickle.dump(z, c)

