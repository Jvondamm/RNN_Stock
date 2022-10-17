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
DAYS = 20
EPOCHS = 50
BATCH_SIZE = 32


def create_datasets(df):
    xUP = []
    yUP = []
    xDOWN = []
    yDOWN = []
    for i in range(50, df.shape[0]):
        avg = sum(df[i-50:i, 0]) / 50
        if df[i,0] <= avg:
            xUP.append(df[i-50:i, 0])
            yUP.append(df[i, 0])
        else:
            xDOWN.append(df[i-50:i, 0])
            yDOWN.append(df[i, 0])

    xUP = np.array(xUP)
    yUP = np.array(yUP)
    xDOWN = np.array(xDOWN)
    yDOWN = np.array(yDOWN)
    return xUP, yUP, xDOWN, yDOWN


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
    print(len(sys.argv))
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python3 rnn.py <ticker> <optional model file>")
        exit(-1)
    elif len(sys.argv) == 3:
        saved = sys.argv[2]

    epochs = EPOCHS
    batch_size = BATCH_SIZE

    t = sys.argv[1]
    ticker = yf.Ticker(t)
    info = None

    try:
        info = ticker.info
    except:
        print("Invalid ticker: " + t)
        print("Exiting...")
        exit(-1)

    if len(sys.argv) == 3:
        PredictPrice(t, epochs, batch_size, saved)
    else:
        PredictPrice(t, epochs, batch_size)

    return 0

def PredictPrice(t, epochs, batch_size, saved=None):
    today = datetime.date.today()
    prediction_date = today - relativedelta(months=4)

    y_df_test = pdr.get_data_yahoo(t, start=prediction_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    days_to_predict = DAYS
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

    x_trainUP, y_trainUP, x_trainDOWN, y_trainDOWN = create_datasets(dataset_train)
    x_test, y_test = create_dataset(dataset_test)

    if saved == None:
        modelUP = Sequential()
        modelUP.add(LSTM(units=96, return_sequences=True, input_shape=(x_trainUP.shape[1], 1)))
        modelUP.add(Dropout(0.2))
        modelUP.add(LSTM(units=96, return_sequences=True))
        modelUP.add(Dropout(0.2))
        modelUP.add(LSTM(units=96, return_sequences=True))
        modelUP.add(Dropout(0.2))
        modelUP.add(LSTM(units=96))
        modelUP.add(Dropout(0.2))
        modelUP.add(Dense(units=1))

        modelDOWN = Sequential()
        modelDOWN.add(LSTM(units=96, return_sequences=True, input_shape=(x_trainDOWN.shape[1], 1)))
        modelDOWN.add(Dropout(0.2))
        modelDOWN.add(LSTM(units=96, return_sequences=True))
        modelDOWN.add(Dropout(0.2))
        modelDOWN.add(LSTM(units=96, return_sequences=True))
        modelDOWN.add(Dropout(0.2))
        modelDOWN.add(LSTM(units=96))
        modelDOWN.add(Dropout(0.2))
        modelDOWN.add(Dense(units=1))

    x_trainUP = np.reshape(x_trainUP, (x_trainUP.shape[0], x_trainUP.shape[1], 1))
    x_trainDOWN = np.reshape(x_trainDOWN, (x_trainDOWN.shape[0], x_trainDOWN.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    if saved == None:
        modelUP.compile(loss='mean_squared_error', optimizer='adam', metrics=[keras.metrics.MeanAbsolutePercentageError()])
        modelUP.fit(x_trainUP, y_trainUP, epochs=epochs, batch_size=batch_size)
        modelUP.save('modelUP')

        modelDOWN.compile(loss='mean_squared_error', optimizer='adam', metrics=[keras.metrics.MeanAbsolutePercentageError()])
        modelDOWN.fit(x_trainDOWN, y_trainDOWN, epochs=epochs, batch_size=batch_size)
        modelDOWN.save('modelDOWN')

    if saved != None:
        modelUP = keras.models.load_model('modelUP')
        modelDOWN = keras.models.load_model('modelDOWN')

    predictions_nUP = []
    predictions_nDOWN = []
    testUP = x_test[-days_to_predict:]
    testDOWN = x_test[-days_to_predict:]
    for i in range(days_to_predict):
        predictions_nUP.append(modelUP.predict(testUP)[0])
        testUP = testUP[0][1:]
        testUP = np.insert(testUP, -1, predictions_nUP[i])
        testUP = testUP.reshape((1, 50, 1))

        predictions_nDOWN.append(modelDOWN.predict(testDOWN)[0])
        testDOWN = testDOWN[0][1:]
        testDOWN = np.insert(testDOWN, -1, predictions_nDOWN[i])
        testDOWN = testDOWN.reshape((1, 50, 1))

    predictions_nUP = scaler.inverse_transform(predictions_nUP)
    predictions_oneUP = modelUP.predict(x_test[-days_to_predict:])
    metricsUP = modelUP.evaluate(x_test, y_test)
    predictions_oneUP = scaler.inverse_transform(predictions_oneUP)
    Plot(t, totalData, metricsUP, prediction_date, days_to_predict, predictions_nUP, predictions_oneUP, "UP")

    predictions_nDOWN = scaler.inverse_transform(predictions_nDOWN)
    predictions_oneDOWN = modelDOWN.predict(x_test[-days_to_predict:])
    metricsDOWN = modelDOWN.evaluate(x_test, y_test)
    predictions_oneDOWN = scaler.inverse_transform(predictions_oneDOWN)
    Plot(t, totalData, metricsDOWN, prediction_date, days_to_predict, predictions_nDOWN, predictions_oneDOWN, "DOWN")

def Plot(t, totalData, metrics, prediction_date, days_to_predict, predictions_n, predictions_one, type="UP"):

    today = datetime.date.today()
    totalDates = pd.date_range(start="2020-01-01", end=today.strftime('%Y-%m-%d'), freq='B')
    predictionDates = pd.date_range(start=prediction_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'), freq='B')
    totalData.reindex(totalDates)

    colors = cycler('color',
                    ['#EE6666', '#3388BB', '#9988DD',
                     '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('xtick', direction='out', color='gray')
    plt.rc('ytick', direction='out', color='gray')
    plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2)
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(totalDates[-int(days_to_predict * 2):], (totalData['Open'])[-int(days_to_predict * 2):], label='Actual Price')
    plt.plot(predictionDates[-len(predictions_n):], predictions_n, label='Predicted Price (n-Step)')
    plt.plot(predictionDates[-len(predictions_one):], predictions_one, label='Predicted Price (1-Step)')
    plt.legend()
    # ax.text(.05, 0.95, f"Mean Absolute % Error: {metrics[1]: .2f}%", verticalalignment='bottom')
    ax.set_ylabel("Stock Price ($ USD)", fontsize=18, color='gray')
    ax.set_xlabel("Date", fontsize=18, color='gray')
    plt.title(t + " Stock Price Prediction", fontsize=25, color='gray')
    if type == "UP":
        plt.savefig('graphUP.png')
    else:
        plt.savefig('graphDOWN.png')
    print(f"Mean Absolute % Error: {metrics[1]: .2f}%")


if __name__ == "__main__":
    main()