import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from pandas_datareader import data as pdr
import sys

yf.pdr_override()

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

    if len(sys.argv) == 1:
        print("Must provide a ticker")
        exit()
    elif len(sys.argv) > 2:
        print("Too many arguments.")
        exit()
    else:
        t = sys.argv[1]

    ticker = yf.Ticker(t)
    info = None

    try:
        info = ticker.info
    except:
        print("Invalid ticker: " + t)
        print("Exiting...")
        exit(-1)
    
    df = pdr.get_data_yahoo(t, start="2010-07-01", end="2021-01-30")

    df = df['Open'].values
    df = df.reshape(-1, 1)

    dataset_train = np.array(df[:int(df.shape[0]*0.8)])
    dataset_test = np.array(df[int(df.shape[0]*0.8):])
    print(dataset_train.shape)
    print(dataset_test.shape)

    scaler = MinMaxScaler(feature_range=(0,1))

    dataset_train = scaler.fit_transform(dataset_train)
    dataset_train[:5]

    dataset_test = scaler.transform(dataset_test)
    dataset_test[:5]

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

    model.fit(x_train, y_train, epochs=50, batch_size=32)
    model.save('stock_prediction4.h5')

    model = load_model('stock_prediction4.h5')

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('#000041')
    ax.plot(y_test_scaled, color='red', label='Original price')
    plt.plot(predictions, color='cyan', label='Predicted price')
    plt.legend()
    plt.savefig('graph.png')

if __name__ == "__main__":
    main()