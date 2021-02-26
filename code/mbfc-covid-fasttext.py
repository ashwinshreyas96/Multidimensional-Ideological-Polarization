import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
import sys
from nltk.tokenize import TweetTokenizer
import preprocessor as p
from nltk.corpus import stopwords
import string
import logging
from gensim.parsing.preprocessing import remove_stopwords
import multiprocessing
import time
import langid
import os
from scipy import stats
from sklearn import preprocessing
from collections import Counter
from functools import reduce
import re
import sent2vec

num_cores = multiprocessing.cpu_count()

model = sent2vec.Sent2vecModel()
model.load_model('../models/twitter_bigrams.bin')

tokenizer = TweetTokenizer()

def deep_clean(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',tweet)
    tweet = re.sub('(\@[^\s]+)','<user>',tweet)
    try:
        tweet = tweet.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    return tweet

def preprocess_tweet(tweet):
    cleaned = p.clean(tweet.lower())
    cleaned = cleaned.strip('/n')
    cleaned = cleaned.replace('[^\w\s]', ' ').replace('\s\s+', ' ')
    cleaned = remove_stopwords(cleaned)
    cleaned = tokenizer.tokenize(cleaned)
    cleaned = deep_clean(' '.join(cleaned))
    return cleaned

def check_lang(tweet):
    lang = langid.classify(tweet)[0]
    return lang

def group_by_user(tweet,screen_name):
    if tweet.startswith('RT @'):
        rt_user = tweet.split('RT @')[1]
        rt_user = rt_user.split(':')[0]
        if rt_user not in content_generated_by_user:
            content_generated_by_user[rt_user] = []
            content_user_count[rt_user]=0
        if rt_user in content_generated_by_user and tweet not in content_generated_by_user[rt_user]:
            content_generated_by_user[rt_user].append(preprocess_tweet(tweet))
            content_user_count[rt_user]+=1
        if screen_name not in content_generated_by_user:
            content_generated_by_user[screen_name] = []
            content_user_count[screen_name]=0
        content_generated_by_user[screen_name].append(preprocess_tweet(tweet))
        content_user_count[screen_name]+=1

    elif not tweet.startswith('RT @') and 'RT @' in tweet:
        rt_user = tweet.split('RT @')[1]
        rt_user = rt_user.split(':')[0]
        if rt_user not in content_generated_by_user:
            content_generated_by_user[rt_user] = []
            content_user_count[rt_user]=0
        if rt_user in content_generated_by_user and tweet not in content_generated_by_user[rt_user]:
            content_generated_by_user[rt_user].append(preprocess_tweet(tweet.split('RT @')[1]))
            content_user_count[rt_user]+=1
        if screen_name not in content_generated_by_user:
            content_generated_by_user[screen_name] = []
            content_user_count[screen_name]=0
        content_generated_by_user[screen_name].append(preprocess_tweet(tweet))
        content_user_count[screen_name]+=1
    else:
        if screen_name not in content_generated_by_user:
            content_generated_by_user[screen_name] = []
            content_user_count[screen_name]=0
        content_generated_by_user[screen_name].append(preprocess_tweet(tweet))
        content_user_count[screen_name]+=1


newdf = pd.read_csv(sys.argv[1]) #Pass a file containing all tweets over time.
newdf['text'] = newdf['text'].astype('str')
newdf = newdf[newdf.text != '']
newdf = newdf[newdf.text.notna()].reset_index()
tqdm.pandas()
newdf['lang'] = newdf['text'].progress_apply(check_lang)
newdf = newdf[newdf['lang']=='en']

content_generated_by_user = {}
content_user_count = {}

for i in tqdm(range(len(newdf))):
    group_by_user(newdf['text'].iloc[i],newdf['screen_name'].iloc[i])

content_df = pd.DataFrame(content_user_count.items(),columns=['user','count'])
content_df.to_pickle('../data/en-content_df.pkl')

content_user_tweets={}

def generate_embeddings(user,content_user_tweets,ind):
    ind[0]+=1
    users_tweets = content_generated_by_user[user]
    users_tweet = '.'.join(users_tweets)
    content_user_tweets[user] = users_tweet

def main():
    manager = multiprocessing.Manager()
    ind = manager.list()
    ind.append(0)
    content_user_tweets = manager.dict()
    pool = multiprocessing.Pool(processes=num_cores-1)
    for user in content_generated_by_user:
        pool.apply_async(generate_embeddings,args=[user,content_user_tweets,ind])
    pool.close()
    pool.join()

    content_user_tweets=dict(content_user_tweets)

    combined_data_df = pd.DataFrame(content_user_tweets.items(),columns=['user','tweets'])
    combined_data_df.to_pickle('../data/combined_data_df.pkl')

    res1 = dict(list(content_user_tweets.items())[len(content_user_tweets)//2:])
    res2 = dict(list(content_user_tweets.items())[:len(content_user_tweets)//2])

    tweets_to_embed = list(res2.values())
    embeddings = model.embed_sentences(tweets_to_embed)

    count=0
    content_user_embeddings={}
    for user in list(res2.keys()):
        content_user_embeddings[user]=embeddings[count]
        count+=1

    embed_df = pd.DataFrame.from_dict(content_user_embeddings, orient='index')
    embed_df.to_pickle('../models/en-all-users-mbfc-fasttext-embed_df_2.pkl')

    tweets_to_embed = list(res1.values())
    embeddings = model.embed_sentences(tweets_to_embed)

    count=0
    content_user_embeddings={}
    for user in res1:
        content_user_embeddings[user]=embeddings[count]
        count+=1

    embed_df = pd.DataFrame.from_dict(content_user_embeddings, orient='index')
    embed_df.to_pickle('../models/en-all-users-mbfc-fasttext-embed_df_1.pkl')

main()
