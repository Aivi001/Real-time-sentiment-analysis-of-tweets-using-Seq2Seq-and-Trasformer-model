import datetime as dt
import re

import pandas as pd
import streamlit as st
from flair.data import Sentence
from flair.models import TextClassifier
import os
import tweepy as tw
import pandas as pd
import torch

consumer_key= 'XXXX'
consumer_secret= 'XXXX'
access_token= 'XXXX'
access_token_secret= 'XXXX'


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Set page title
st.title('Twitter Sentiment Analysis')

# Load classification model
with st.spinner('Loading classification model...'):
    
    classifier = TextClassifier.load('model-saves/my_fine_tuned_bert1.pt')

# Preprocess function
allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
punct = '!?,.@#'
maxlen = 280

def preprocess(text):
    # Delete URLs, cut to maxlen, space out punction with spaces, and remove unallowed chars
    return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE) if char in allowed_chars]])


### TWEET SEARCH AND CLASSIFY ###
st.subheader('Search Twitter for Query')

# Get user input
query = st.text_input('Query:', '#')

# As long as the query is valid (not empty or equal to '#')...
if query != '' and query != '#':
    with st.spinner('Searching for and analyzing {query}...'):
        
        search_words = query 
        date_since = "2020-05-16"

        # Collect tweets
        tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since).items(5)

        
        # Initialize empty dataframe
        tweet_data = pd.DataFrame({
            'tweet': [],
            'predicted-sentiment': []
        })

        # Keep track of positive vs. negative tweets
        pos_vs_neg = {'0' : 0, '1' : 0}

        # Add data for each tweet
        for tweet in tweets:
            # Skip iteration if tweet is empty
            if tweet.text in ('', ' '):
                continue
            # Make predictions
            sentence = Sentence(preprocess(tweet.text))

            classifier.predict(sentence)
            sentiment = sentence.labels[0]
            sentiment.value = '1' if sentiment.value == '4' else '0'
             
            # Keep track of positive vs. negative tweets
            pos_vs_neg[sentiment.value] += 1
            # Append new data
            tweet_data = tweet_data.append({'tweet': tweet.text, 'predicted-sentiment': sentiment.value }, ignore_index=True)

# Show query data and sentiment if available
try:
    st.write(tweet_data)
    try:
        if pos_vs_neg['1'] > pos_vs_neg['0']:
            st.write('Positive response')
        else:
            st.write('Negative response')
    except ZeroDivisionError: # if no negative tweets
        st.write('All postive tweets')
except NameError: # if no queries have been made yet
    pass
