import keras.metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from pandas_datareader import data as pdr
import sys
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import os

yf.pdr_override()
IPython_default = plt.rcParams.copy()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y

def main():

    if len(sys.argv) != 5 and len(sys.argv) != 3:
        print("Usage: python3 rnn.py <ticker> <integer of months to predict> <optional epochs> <optional batch size>")
        exit(-1)
    elif len(sys.argv) == 5:
        epochs = int(sys.argv[3])
        batch_size = int(sys.argv[4])
    else:
        epochs = 50
        batch_size = 32

    t = sys.argv[1]
    months_to_predict = int(sys.argv[2]) + 3
    ticker = yf.Ticker(t)
    info = None

    try:
        info = ticker.info
    except:
        print("Invalid ticker: " + t)
        print("Exiting...")
        exit(-1)

    PredictPrice(months_to_predict, t, epochs, batch_size)
    # PredictReturn(months_to_predict, t, epochs, batch_size)
    return 0

def PredictPrice(months_to_predict, t, epochs, batch_size):
    today = datetime.date.today()
    prediction_date = today - relativedelta(months=months_to_predict)

    y_df_test = pdr.get_data_yahoo(t, start=prediction_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    days_to_predict = len(y_df_test)
    y_df_train = pdr.get_data_yahoo(t, start="2020-01-01", end=prediction_date.strftime('%Y-%m-%d'))

    df_test = y_df_test['Open'].values
    df_train = y_df_train['Open'].values

    totalData = pd.concat([y_df_train, y_df_test], axis=0)
    totalData.drop(["Close", "High", "Low", "Volume"], inplace=True, axis=1)

    df_test = df_test.reshape(-1, 1)
    df_train = df_train.reshape(-1, 1)

    dataset_train = np.array(df_train)
    dataset_test = np.array(df_test)

    scaler = MinMaxScaler(feature_range=(0, 1))
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

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[keras.metrics.MeanAbsolutePercentageError()])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    predictions_n = []
    test = np.array(x_test[49]).reshape((1, 50, 1))
    for i in range(days_to_predict - 50):
        predictions_n.append(model.predict(test)[0])
        test = test[0][1:]
        test = np.insert(test, -1, predictions_n[i])
        test = test.reshape((1, 50, 1))

    predictions_n = scaler.inverse_transform(predictions_n)
    predictions_one = model.predict(x_test)
    metrics = model.evaluate(x_test, y_test)
    predictions_one = scaler.inverse_transform(predictions_one)

    totalDates = pd.date_range(start="2020-01-01", end=today.strftime('%Y-%m-%d'), freq='B')
    predictionDates = pd.date_range(start=prediction_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'), freq='B')
    totalData.reindex(totalDates)

    colors = cycler('color',
                    ['#EE6666', '#3388BB', '#9988DD',
                     '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('xtick', direction='out', color='gray')
    plt.rc('ytick', direction='out', color='gray')
    plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(totalDates[-int(days_to_predict * 1.3):], (totalData['Open'])[-int(days_to_predict * 1.3):], label='Actual Price')
    plt.plot(predictionDates[-len(predictions_n):], predictions_n, label='Predicted Price (n-Step)')
    plt.plot(predictionDates[-len(predictions_one):], predictions_one, label='Predicted Price (1-Step)')
    plt.legend()
    # ax.text(.05, 0.95, f"Mean Absolute % Error: {metrics[1]: .2f}%", verticalalignment='bottom')
    ax.set_ylabel("Stock Price ($ USD)", fontsize=18, color='gray')
    ax.set_xlabel("Date", fontsize=18, color='gray')
    plt.title(t + " Stock Price Prediction", fontsize=25, color='gray')
    plt.savefig('graph.png')
    print(f"Mean Absolute % Error: {metrics[1]: .2f}%")

if __name__ == "__main__":
    main()