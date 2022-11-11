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
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import praw
import pandas as pd
import plotly.express as px

yf.pdr_override()
IPython_default = plt.rcParams.copy()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DAYS = 20               # Days to predict
EPOCHS = 50             # Model training runs
BATCH_SIZE = 32         # Model size per epoch
SAVE_REDDIT_CSV = False # Saves Reddit Post stats and sentiment into csv
SUBREDDITS = ["Investing",
    "Stocks",
    "Economics",
    "StockMarket",
    "Economy",
    "GlobalMarkets",
    "WallStreetBets",
    "Options",
    "Finance",
    "Bitcoin",
    "Dividends",
    "Cryptocurrency",
    "SecurityAnalysis",
    "AlgoTrading",
    "DayTrading"]       # Subreddits to webscrape


# The sentiment value is public's opinion of the stock between -1 to 1.
# The df input data is normalized between 0 and 1.
# I get the previous 50 days and get their avg, then put it into a df with the "key" being:
# the 51st day - avg - sentiment. So this gets the difference between the 51st day price and the avg,
# then the difference of THAT and the sentiment. Then I sort by that "key"
#
# Then I essentially split my df into 20 sections, and grab one section based on my sentiment value.
# For example if sentiment is .5, I grab the 15th section. If -.5, I grab the 5th section.
#
# This is the data I train my model on, therefore matching the price trend to the current stock sentiment.
def createSentimentDataset(df, sentiment):
    x = []
    y = []

    halfLength = len(df) / 2
    df_stats = [[0]*(df.shape[0]),[0]*(df.shape[0])]

    for i in range(50, df.shape[0]):
        avg = sum(df[i-50:i, 0]) / 50
        df_stats[0][i] = (df[i,0] - avg) - sentiment
        df_stats[1][i] = i

    df_stats = sorted(df_stats, key=lambda x: x[0])

    if sentiment > 0:
        upper = int(sentiment * halfLength + halfLength)
        lower = int(upper - (halfLength / 10))
    else:
        upper = int(sentiment * halfLength)
        lower = int(upper - (halfLength / 10))

    for i in range(lower, upper):
        x.append(df[df_stats[1][i]-50:df_stats[1][i], 0])
        y.append(df[df_stats[1][i], 0])

    x = np.array(x)
    y = np.array(y)
    return x, y


# Creates a dataset in the form of
# y: [0:50]
# x: [51]
# then increases indices by 1.
def createDataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


# Returns average sentiment for each comment of a reddit post.
def commentSentiment(reddit, urlT):
    subComments = []
    bodyComment = []
    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 0

    for comment in subComments:
        try:
            bodyComment.append(comment.body)
        except:
            return 0

    sia = SIA()
    results = []
    for line in bodyComment:
        scores = sia.polarity_scores(line)
        scores['headline'] = line

        results.append(scores)

    df =pd.DataFrame.from_records(results)
    df.head()
    df['label'] = 0

    try:
        df.loc[df['compound'] > 0.1, 'label'] = 1
        df.loc[df['compound'] < -0.1, 'label'] = -1
    except:
        return 0

    averageScore = 0
    position = 0
    while position < len(df.label)-1:
        averageScore = averageScore + df.label[position]
        position += 1
    averageScore = averageScore/len(df.label)

    return(averageScore)


# Returns latest comment date of a post
def latestComment(reddit, urlT):
    subComments = []
    updateDates = []
    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 0

    for comment in subComments:
        try:
            updateDates.append(comment.created_utc)
        except:
            return 0

    updateDates.sort()
    return(updateDates[-1])


# Generates sentiment value of stock ticker via web scraping Reddit
# then applying vadars' lexicon database
def getSentiment(ticker):
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

    reddit = praw.Reddit(client_id='0QJCf-sHV__XlNXBnviE6g',
                        client_secret='7-0NJ5QigYaKQCvUsnCMRfwcOQnRww',
                        user_agent='virxxd')

    subreddits = SUBREDDITS
    submission_statistics = []
    d = {}

    for subreddit in subreddits:
        for submission in reddit.subreddit(subreddit).search(ticker, limit=130):
            past = datetime.datetime.now() - datetime.timedelta(days=30)
            date = datetime.datetime.fromtimestamp(submission.created_utc)
            if past > date:
                continue
            d = {}
            d['ticker'] = ticker
            d['num_comments'] = submission.num_comments
            d['comment_sentiment_average'] = commentSentiment(reddit, submission.url)
            if d['comment_sentiment_average'] == 0.000000:
                continue
            d['latest_comment_date'] = latestComment(reddit, submission.url)
            d['score'] = submission.score
            d['upvote_ratio'] = submission.upvote_ratio
            d['date'] = submission.created_utc
            d['domain'] = submission.domain
            d['num_crossposts'] = submission.num_crossposts
            d['author'] = submission.author
            submission_statistics.append(d)

    dfSentimentStocks = pd.DataFrame(submission_statistics)

    _timestampcreated = dfSentimentStocks["date"].apply(getDate)
    dfSentimentStocks = dfSentimentStocks.assign(timestamp = _timestampcreated)

    _timestampcomment = dfSentimentStocks["latest_comment_date"].apply(getDate)
    dfSentimentStocks = dfSentimentStocks.assign(commentdate = _timestampcomment)

    dfSentimentStocks.sort_values("latest_comment_date", axis = 0, ascending = True,inplace = True, na_position ='last')

    dfSentimentStocks.author.value_counts()

    if SAVE_REDDIT_CSV:
        dfSentimentStocks.to_csv('Reddit_Sentiment_Equity.csv', index=False)

    print("Overall Sentiment: ", dfSentimentStocks['comment_sentiment_average'].sum() / len(dfSentimentStocks.index))
    return dfSentimentStocks['comment_sentiment_average'].sum() / len(dfSentimentStocks.index)


# Returns reddit post date
def getDate(date):
    return datetime.datetime.fromtimestamp(date)


# Takes sentiment value, fetches stock price history, generates control and actual LSTM models,
# trains and predicts future prices using models, then calls plot()
def predictPrice(t, epochs, batch_size, sentiment, saved=None):
    today = datetime.date.today()
    prediction_date = today - relativedelta(months=4)

    y_df_test = pdr.get_data_yahoo(t, start=prediction_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    days_to_predict = DAYS
    y_df_train = pdr.get_data_yahoo(t, start="2000-01-01", end=prediction_date.strftime('%Y-%m-%d'))

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

    x_test, y_test = createDataset(dataset_test)

    if saved == None:
        x_train, y_train = createDataset(dataset_train)
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

        x_sent_train, y_sent_train = createSentimentDataset(dataset_train, sentiment)
        modelS = Sequential()
        modelS.add(LSTM(units=96, return_sequences=True, input_shape=(x_sent_train.shape[1], 1)))
        modelS.add(Dropout(0.2))
        modelS.add(LSTM(units=96, return_sequences=True))
        modelS.add(Dropout(0.2))
        modelS.add(LSTM(units=96, return_sequences=True))
        modelS.add(Dropout(0.2))
        modelS.add(LSTM(units=96))
        modelS.add(Dropout(0.2))
        modelS.add(Dense(units=1))

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    if saved == None:
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[keras.metrics.MeanAbsolutePercentageError()])
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        model.save('model2')

        x_sent_train = np.reshape(x_sent_train, (x_sent_train.shape[0], x_sent_train.shape[1], 1))
        modelS.compile(loss='mean_squared_error', optimizer='adam', metrics=[keras.metrics.MeanAbsolutePercentageError()])
        modelS.fit(x_sent_train, y_sent_train, epochs=epochs, batch_size=batch_size)
        modelS.save('model2S')

    if saved != None:
        modelS = keras.models.load_model('model2')
        modelS = keras.models.load_model('model2S')

    predictions_n = []
    test = x_test
    for i in range(days_to_predict):
        predictions_n.append(modelS.predict(test)[0])
        test = test[0][1:]
        test = np.insert(test, -1, predictions_n[i])
        test = test.reshape((1, 50, 1))

    predictions_n = scaler.inverse_transform(predictions_n)
    predictions_one = model.predict(x_test[-days_to_predict:])
    metrics = model.evaluate(x_test, y_test)
    predictions_one = scaler.inverse_transform(predictions_one)
    plot(t, totalData, metrics, prediction_date, days_to_predict, predictions_n, predictions_one)
    plot2(t, totalData, metrics, prediction_date, days_to_predict, predictions_n, predictions_one)

# Plots ontrol and actual predictions according to real data
def plot(t, totalData, metrics, prediction_date, days_to_predict, predictions_n, predictions_one):

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
    ax.set_ylabel("Stock Price ($ USD)", fontsize=18, color='gray')
    ax.set_xlabel("Date", fontsize=18, color='gray')
    plt.title(t + " Stock Price Prediction", fontsize=25, color='gray')
    plt.savefig('graph2.png')
    print(f"Mean Absolute % Error: {metrics[1]: .2f}%")


def plot2(t, totalData, metrics, prediction_date, days_to_predict, predictions_n, predictions_one):
    today = datetime.date.today()
    totalDates = pd.date_range(start="2020-01-01", end=today.strftime('%Y-%m-%d'), freq='B')
    predictionDates = pd.date_range(start=prediction_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'), freq='B')
    totalData.reindex(totalDates)
    df = pd.DataFrame(
        totalDates[-int(days_to_predict * 2):],
        (totalData['Open'])[-int(days_to_predict * 2):]
        )
    fig = px.line(df, title='Actual Price')
    fig.show()


# Main
def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python3 rnn2.py <ticker> <optional model file>")
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
        predictPrice(t, epochs, batch_size, 0, saved)
    else:
        # sentiment = getSentiment(ticker)
        sentiment = -1
        predictPrice(t, epochs, batch_size, sentiment)
    return 0


if __name__ == "__main__":
    main()