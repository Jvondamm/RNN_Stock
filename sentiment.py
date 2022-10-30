# SOURCE: https://github.com/tstewart161/Reddit_Sentiment_Trader/blob/main/main.py

# In[1]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import praw
import matplotlib.pyplot as plt
import math
import datetime as dt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# In[2]:


nltk.download('vader_lexicon')
nltk.download('stopwords')


# In[3]:


reddit = praw.Reddit(client_id='0QJCf-sHV__XlNXBnviE6g',
                    client_secret='7-0NJ5QigYaKQCvUsnCMRfwcOQnRww',
                    user_agent='virxxd')


# In[4]:


subreddits = ["Investing",
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
"DayTrading"]
stocks = ["AMZN"]

# In[5]:


def commentSentiment(ticker, urlT):
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


# In[6]:


def latestComment(ticker, urlT):
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


# In[7]:


def get_date(date):
    return dt.datetime.fromtimestamp(date)


# In[8]:


submission_statistics = []
d = {}
for ticker in stocks:
    for subreddit in subreddits:
        for submission in reddit.subreddit(subreddit).search(ticker, limit=130):
            past = datetime.now() - timedelta(days=30)
            date = datetime.fromtimestamp(submission.created_utc)
            if past > date:
                continue
            d = {}
            d['ticker'] = ticker
            d['num_comments'] = submission.num_comments
            d['comment_sentiment_average'] = commentSentiment(ticker, submission.url)
            if d['comment_sentiment_average'] == 0.000000:
                continue
            d['latest_comment_date'] = latestComment(ticker, submission.url)
            d['score'] = submission.score
            d['upvote_ratio'] = submission.upvote_ratio
            d['date'] = submission.created_utc
            d['domain'] = submission.domain
            d['num_crossposts'] = submission.num_crossposts
            d['author'] = submission.author
            submission_statistics.append(d)

dfSentimentStocks = pd.DataFrame(submission_statistics)

_timestampcreated = dfSentimentStocks["date"].apply(get_date)
dfSentimentStocks = dfSentimentStocks.assign(timestamp = _timestampcreated)

_timestampcomment = dfSentimentStocks["latest_comment_date"].apply(get_date)
dfSentimentStocks = dfSentimentStocks.assign(commentdate = _timestampcomment)

dfSentimentStocks.sort_values("latest_comment_date", axis = 0, ascending = True,inplace = True, na_position ='last')

dfSentimentStocks


# In[9]:


dfSentimentStocks.author.value_counts()


# In[10]:


dfSentimentStocks.to_csv('Reddit_Sentiment_Equity.csv', index=False)

print(dfSentimentStocks['comment_sentiment_average'].sum() / len(dfSentimentStocks.index))