import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from pandas_datareader import data as pdr
import sys
import datetime
from matplotlib.ticker import Formatter
from matplotlib import dates

yf.pdr_override()

# TODO
# format prediction time
# format plots
# make algorithm better

def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

def main():

    if len(sys.argv) != 4 and len(sys.argv != 2):
        print("Usage: python3 rnn.py <ticker> <integer of months to predict> <optional epochs> <optional batch size>")
        exit(-1)
    elif len(sys.argv == 4):
        epochs = sys.argv[2]
        batch_size = sys.argv[3]
    else:
        epochs = 50
        batch_size = 32

    t = sys.argv[1]
    length_to_predict = sys.argv[2] * 31
    ticker = yf.Ticker(t)
    info = None

    try:
        info = ticker.info
    except:
        print("Invalid ticker: " + t)
        print("Exiting...")
        exit(-1)

    today = datetime.date.today()
    prediction_date = today - datetime.timedelta(int(length_to_predict))

    df_test = pdr.get_data_yahoo(t, start=prediction_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    df_train = pdr.get_data_yahoo(t, start="2010-07-01", end=prediction_date.strftime('%Y-%m-%d'))
    df_test = df_test['Open'].values
    df_train = df_train['Open'].values

    df_test = df_test.reshape(-1, 1)
    df_train = df_train.reshape(-1, 1)

    dataset_train = np.array(df_train)
    dataset_test = np.array(df_test)

    scaler = MinMaxScaler(feature_range=(0,1))

    dataset_train = scaler.fit_transform(dataset_train)
    dataset_test = scaler.transform(dataset_test)

    x_train, y_train = create_dataset(dataset_train)
    x_test, y_test = create_dataset(dataset_test)

    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_facecolor('#000041')
    formatter = MyFormatter(predictions)
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(y_test_scaled, color='red', label='Original price')
    plt.plot(predictions, color='cyan', label='Predicted price')
    plt.legend()
    plt.savefig('graph.png')


class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        """Return the label for time x at position pos."""
        ind = int(round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''
        return dates.num2date(self.dates[ind]).strftime(self.fmt)



if __name__ == "__main__":
    main()