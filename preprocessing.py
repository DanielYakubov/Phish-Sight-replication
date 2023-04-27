import numpy as np
import re
import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from urllib.parse import urlparse


#read and get the data
def get_data(data_path):
    df = pd.read_csv(data_path)
    return df

#removing null and duplicate values
def drop_nulls_and_duplicates(df):
    df=df.dropna().reset_index(drop=True)
    # df=df.drop_duplicates(keep='first').reset_index(drop=True)
    df=df.drop_duplicates(subset='URL', keep="first").reset_index(drop=True)
    return df


# #scaling of the colors - not needed now
# def color_scaling(df):
#     for i in range(len(df['red'])):
#         df['red'][i] = df['red'][i] / 255
#     for i in range(len(df['green'])):
#         df['green'][i] = df['green'][i] / 255
#     for i in range(len(df['blue'])):
#         df['blue'][i] = df['blue'][i] / 255
#     return df


#filter out texts that are not repeating in the dataset
def non_unique_text(df):
    ty=pd.DataFrame(data=df['text'].value_counts().sort_values(ascending=False))
    ty=ty.loc[ty['text'] != 1]

    rem_index = []
    for i in range(len(ty)):
        common_loc = df.loc[df['text'] == ty.index[i]]
        for j in range(len(df)):
            if df['text'][j] == ty.index[i]:
                rem_index.append(j)
        del common_loc 

    df=df.drop(labels=rem_index,axis=0)
    df=df.reset_index(drop=True)
    return df


#removing data points whose url's do not exist or are non-functional
def drop_bad_pages(df):
    bad_pages = []
    bad_pattern = r"Site Not Found|This page isn’t working|Internal Server Error|This page isn’t working|Your connection is not private|404"
    for i in range(len(df)):
        if re.match(bad_pattern, df['text'][i], flags=re.IGNORECASE):
            bad_pages.append(i)
    df=df.drop(labels=bad_pages,axis=0)
    df=df.reset_index(drop=True)
    return df


#conversion of brands to numericals
def one_hot_encoding(df):
    one_hot = pd.get_dummies(df.brand_name)
    df = pd.concat([df, one_hot], axis=1)
    return df


#get the domain name
def get_top_level_domain(url):
    tld = urlparse(url)
    return tld.netloc


#get the domain name's length
def get_url_len(url_str):
    return len(url_str)


#get n-grams for each text(in our case it is n=2)
def get_english_char_bigram_probs():
    words = nltk.corpus.words.words('en') # loading in a corpus of words
    bg_cnts = {}
    ung_cnts = {}
    # going through each word in the corpus
    for word in words:
        padding = ' ' # needed for bigram models
        word = padding + word.lower() + padding
        # getting characters
        chars = [c for c in word]
        bgs = nltk.bigrams(chars) # getting bigrams
        
        # adding bigram to counts dictionary
        for bigram in bgs:
            bg_cnts[bigram] = bg_cnts[bigram] + 1 if bigram in bg_cnts else 1
            
        #adding unigram to count dictionary
        for unigram in chars:
            ung_cnts[unigram] = ung_cnts[unigram] + 1 if unigram in ung_cnts else 1
    
    # turning each count into a probability (MLE)
    for k, v in bg_cnts.items():
        first_word, _ = k
        denom = ung_cnts[first_word]
        bg_cnts[k] = v/denom # now it is a conditional probability
        
    # same for unigrams
    norm = sum(ung_cnts.values())
    for k, v in ung_cnts.items():
        ung_cnts[k] = v/norm

    return bg_cnts, ung_cnts


#score of each url texts' bigrams
def get_score(ung_probs, bg_probs, url_str):
    url_chars = [c for c in url_str]
    url_bgs = nltk.bigrams(url_chars)
    
    # getting a score of each url char bigram
    score = 0
    for bigram in url_bgs:
        # using linear interpolatation smoothing in negative log space to prevent underflow (lambda 0.5)
        first_word, _ = bigram
        score += 0.5*bg_probs.get(bigram, 0) + 0.5*ung_probs.get(first_word, 0)
        # TODO probably should do entropy here
    return score


#normal splitting of the dataset into training and testing
def split_data(df):
    y = df['status']
    X = df.drop(['status'], axis=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=42)
    return X_tr, X_te, y_tr, y_te


#store csv's
def store_datasets(Xtrain,Xtest,ytrain,ytest):
    Xtrain.to_csv('data/X_train.csv',index=False)
    Xtest.to_csv('data/X_test.csv',index=False)
    ytrain.to_csv('data/y_train.csv',index=False)
    ytest.to_csv('data/y_test.csv',index=False)


if __name__ == "__main__":
    data = get_data("data/scraped/all_data_scraped.csv")
    # data['status'] = np.where(data['status']=="legitimate",0,1)
    data = drop_nulls_and_duplicates(data)
    # data = color_scaling(data)
    data = non_unique_text(data)
    data = drop_bad_pages(data)
    data = one_hot_encoding(data)
    data = data.drop(['brand_name', 'no_brand'], axis=1)

    bg_probs, ung_probs = get_english_char_bigram_probs()
    url_lens = []
    url_scores = []
    for url in data['URL']:
        tld = get_top_level_domain(url)
        l = get_url_len(tld)
        score = get_score(ung_probs, bg_probs, tld)
        url_lens.append(l)
        url_scores.append(score)

    # adding new features to data
    data['url_len'] = url_lens
    data['tld_char_score'] = url_scores

    # # needed for redo
    # data[["URL", "status"]].to_csv('data/links/links_for_redo.csv', index=False)

    # dropping all text features
    data = data.drop(["URL", "text"], axis=1)

    X_train, X_test, y_train, y_test = split_data(data)
    store_datasets(X_train, X_test, y_train, y_test)