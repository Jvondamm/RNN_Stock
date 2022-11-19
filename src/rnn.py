import os
import argparse
import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import keras.metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler
import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import praw
import config

# suppress tf errors
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

yf.pdr_override()
IPython_default = plt.rcParams.copy()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DAYS = 20               # Days to predict
POST_DAYS = 30          # Max Reddit posts scraped
EPOCHS = 50             # Model training runs
BATCH_SIZE = 32         # Model size per epoch
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
    "DayTrading",
    "PersonalFinance",
    "Investing",
    "Dividends",
    "StocksToBuyToday",
    "Politics",
    "WorldNews"]       # Subreddits to webscrape


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
def getSentiment(ticker, csv):

    if csv:
        dfSentimentStocks = pd.read_csv('artifacts/sentiment.csv', index=False)
    else:

        nltk.download('vader_lexicon')
        nltk.download('stopwords')

        reddit = praw.Reddit(client_id=config.client_id,
                            client_secret=config.client_secret,
                            user_agent=config.user_agent)

        subreddits = SUBREDDITS
        submission_statistics = []
        d = {}

        for subreddit in subreddits:
            print("Scraping subreddit: %s" % subreddit)
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
        if dfSentimentStocks.empty:
            print("Could not locate any posts regarding ticker %s, exiting..." % ticker)
            exit(0)

        _timestampcreated = dfSentimentStocks["date"].apply(getDate)
        dfSentimentStocks = dfSentimentStocks.assign(timestamp = _timestampcreated)

        _timestampcomment = dfSentimentStocks["latest_comment_date"].apply(getDate)
        dfSentimentStocks = dfSentimentStocks.assign(commentdate = _timestampcomment)

        dfSentimentStocks.sort_values("latest_comment_date", axis = 0,
        ascending = True,inplace = True, na_position ='last')

        dfSentimentStocks.author.value_counts()
        dfSentimentStocks.to_csv('artifacts/sentiment.csv', index=False)

    print("Parsed %s hits" % str(len(dfSentimentStocks)))
    print("Overall Sentiment: ", dfSentimentStocks['comment_sentiment_average'].sum()
    / len(dfSentimentStocks.index))
    return dfSentimentStocks['comment_sentiment_average'].sum() / len(dfSentimentStocks.index)


# Returns reddit post date
def getDate(date):
    return datetime.datetime.fromtimestamp(date)


# Takes sentiment value, fetches stock price history, generates control and actual LSTM models,
# trains and predicts future prices using models, then calls plot()
def predictPrice(t, epochs, batch_size, sentiment, model_n, model_one):

    if model_n is not None:
        print("Loading model_n")
        model_n = keras.models.load_model(model_n)
    if model_one is not None:
        print("Loading model_one")
        model_one = keras.models.load_model(model_one)

    # grab beginning-ending dates
    today = datetime.date.today()
    prediction_date = today - relativedelta(months=4)

    y_df_test = pdr.get_data_yahoo(t, start=prediction_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    days_to_predict = DAYS
    y_df_train = pdr.get_data_yahoo(t, start="2000-01-01", end=prediction_date.strftime('%Y-%m-%d'))

    # split training/testing data
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
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    if model_one == None:
        x_train, y_train = createDataset(dataset_train)
        model_one = Sequential()
        model_one.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model_one.add(Dropout(0.2))
        model_one.add(LSTM(units=96, return_sequences=True))
        model_one.add(Dropout(0.2))
        model_one.add(LSTM(units=96, return_sequences=True))
        model_one.add(Dropout(0.2))
        model_one.add(LSTM(units=96))
        model_one.add(Dropout(0.2))
        model_one.add(Dense(units=1))

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model_one.compile(loss='mean_squared_error', optimizer='adam', metrics=[keras.metrics.MeanAbsolutePercentageError()])
        model_one.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        model_one.save('artifacts/model_one')

    if model_n == None:
        x_sent_train, y_sent_train = createSentimentDataset(dataset_train, sentiment)
        model_n = Sequential()
        model_n.add(LSTM(units=96, return_sequences=True, input_shape=(x_sent_train.shape[1], 1)))
        model_n.add(Dropout(0.2))
        model_n.add(LSTM(units=96, return_sequences=True))
        model_n.add(Dropout(0.2))
        model_n.add(LSTM(units=96, return_sequences=True))
        model_n.add(Dropout(0.2))
        model_n.add(LSTM(units=96))
        model_n.add(Dropout(0.2))
        model_n.add(Dense(units=1))

        x_sent_train = np.reshape(x_sent_train, (x_sent_train.shape[0], x_sent_train.shape[1], 1))
        model_n.compile(loss='mean_squared_error', optimizer='adam', metrics=[keras.metrics.MeanAbsolutePercentageError()])
        model_n.fit(x_sent_train, y_sent_train, epochs=epochs, batch_size=batch_size)
        model_n.save('artifacts/model_n')

    predictions_n = []
    test = x_test
    for i in range(days_to_predict):
        predictions_n.append(model_n.predict(test)[0])
        test = test[0][1:]
        test = np.insert(test, -1, predictions_n[i])
        test = test.reshape((1, 50, 1))

    print("Generating Predictions...")
    predictions_n = scaler.inverse_transform(predictions_n)
    predictions_one = model_one.predict(x_test[-days_to_predict:])
    metrics = model_n.evaluate(x_test, y_test)
    predictions_one = scaler.inverse_transform(predictions_one)

    print("Plotting...")
    plot(t, totalData, metrics, prediction_date, days_to_predict, predictions_n, predictions_one)

# Plots control and actual predictions according to real data
def plot(t, totalData, metrics, prediction_date, days_to_predict, predictions_n, predictions_one):

    today = datetime.date.today()
    totalDates = pd.date_range(start="2020-01-01", end=today.strftime('%Y-%m-%d'), freq='B')
    predictionDates = pd.date_range(start=prediction_date.strftime('%Y-%m-%d'),
    end=today.strftime('%Y-%m-%d'), freq='B')
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
    plt.savefig('artifacts/graph.png')
    print(f"Mean Absolute % Error: {metrics[1]: .2f}%")


# Main
def main():
    parser = argparse.ArgumentParser(description='Generate price prediction graph for selected company.')
    parser.add_argument('ticker', help='company stock symbol')
    parser.add_argument('--modelN', dest='model_n', nargs='?', help='path to trained model_n')
    parser.add_argument('--model1', dest='model_one', nargs='?', help='path to trained model_one')
    args = parser.parse_args()

    epochs = EPOCHS
    batch_size = BATCH_SIZE

    ticker = yf.Ticker(args.ticker)
    info = None

    try:
        info = ticker.info
    except:
        print("Invalid ticker: " + args.ticker)
        exit(-1)

    sentiment = getSentiment(args.ticker, args.sentiment)
    predictPrice(args.ticker,
                epochs,
                batch_size,
                sentiment,
                args.model_n,
                args.model_one)
    return 0


if __name__ == "__main__":
    main()