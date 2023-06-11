import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.layers import Layer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# import warnings
# warnings.filterwarnings(action = 'ignore')


#Preprocessing function
def preprocessing(data_frame):
    ## Preprocessing
    # Removing URLs whithin the tweets
    data_frame["Text"] = data_frame["Text"].str.replace(r'\s*https?://\S+(\s+|$)', ' ').str.strip()
    # Removing emails, hashtags and punctuations
    data_frame['Text'] = data_frame["Text"].str.replace(r'\S*@\S*\s?', ' ').str.strip()
    data_frame['Text'] = data_frame['Text'].str.replace(r'#\S*\s?', ' ').str.strip()
    data_frame['Text'] = data_frame['Text'].str.replace(r'[^\w\s]+', ' ').str.strip()

    # Lowercase Text
    data_frame['Text'] = data_frame['Text'].str.lower()

    # # Removing stopwords
    stop = stopwords.words('english')
    data_frame['Text'].apply(lambda x: [item for item in str(x) if item not in stop])

    # Removing newline characters
    data_frame['Text'] = data_frame['Text'].str.rstrip()

    # Tokenizing Posts and counting the length of each post
    data_frame['Tokens'] = data_frame.apply(lambda row: word_tokenize(str(row['Text'])), axis=1)
    data_frame['Length'] = data_frame.apply(lambda row: len(row['Tokens']), axis=1)

    return data_frame

def read_twitter_10000(length = -1):
    ## Preparing the data
    ## Twitter 10000
    Twitter_path = "./Datasets/twitter-suicidal_data_10000.csv"
    if length > 0:
        df = pd.read_csv(Twitter_path, encoding='latin-1', nrows = length)
    elif length == -1:
        df = pd.read_csv(Twitter_path, encoding='latin-1')
    else:
        print("Please choose a proper length")
        exit()

    df = df.rename(columns={'tweet': 'Text', 'intention': 'Label'})
    df = preprocessing(df)
    return df

def read_twitter_tendency(length = -1):
    twitter_path = "./Datasets/suicidal-tendency-tweets.csv"  ## positive samples
    if length > 0:
        df = pd.read_csv(twitter_path, encoding='latin-1', usecols=['tweet', 'intention'], nrows=length)
    elif length == -1:
        df = pd.read_csv(twitter_path, encoding='latin-1', usecols=['tweet', 'intention'], nrows=17142)
    else:
        print("Please choose a proper length")
        exit()

    df = df.rename(columns={'tweet': 'Text', 'intention': 'Label'})
    df = preprocessing(df)
    return df

def read_reddit_SNS(length = -1):
    ## Preparing the data
    Reddit_path = "./Datasets/Suicide_Detection.csv"
    if length > 0:
        df = pd.read_csv(Reddit_path, encoding='latin-1', usecols=['text', 'class'], nrows=length)
    elif length == -1:
        df = pd.read_csv(Reddit_path, encoding='latin-1', usecols=['text', 'class'])
    else:
        print("Please choose a proper length")
        exit()

    df = df.rename(columns={'text': 'Text', 'class': 'Label'})
    df = preprocessing(df)

    label_dict = {'suicide': 1, 'non-suicide': 0}
    df['Label'] = df['Label'].apply(lambda row: label_dict[row])
    return df

def read_reddit_SD(length = -1):
    ## Preparing the data
    reddit_path = "./Datasets/reddit_depression_suicidewatch.csv"
    if length >0:
        df = pd.read_csv(reddit_path, encoding='latin-1', nrows=length)
    elif length == -1:
        df = pd.read_csv(reddit_path, encoding='latin-1')
    else:
        print("Please choose a proper length")
        exit()

    df = df.rename(columns={'text': 'Text', 'label': 'Label'})
    label_dict = {'depression': 0, 'SuicideWatch': 1}
    df['Label'] = df['Label'].apply(lambda row: label_dict[row])
    df = preprocessing(df)
    return df
