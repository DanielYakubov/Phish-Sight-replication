"""This module contains the functions and the code to preprocess the dataset"""

import re
from typing import Dict, Tuple
from urllib.parse import urlparse

import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def drop_nulls_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes nulls and duplicates from the df

    Args:
        df (pd.DataFrame): a generic pandas dataframe

    Returns:
        (pd.DataFrame) that has nulls and duplicates removed
    """
    df = df.dropna().reset_index(drop=True)
    df = df.drop_duplicates(subset='URL', keep="first").reset_index(drop=True)
    return df


def drop_non_unique_text(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out texts that are repeating in the dataset

    Args:
        df (pd.DataFrame): a generic pandas dataframe containing a column texts

    Returns:
        (pd.DataFrame) with unique values
    """
    ty = pd.DataFrame(data=df['text'].value_counts().sort_values(
        ascending=False))
    ty = ty.loc[ty['text'] != 1]

    rem_index = []
    for i in range(len(ty)):
        for j in range(len(df)):
            if df['text'][j] == ty.index[i]:
                rem_index.append(j)

    df = df.drop(labels=rem_index, axis=0)
    df = df.reset_index(drop=True)
    return df


def drop_bad_pages(df: pd.DataFrame) -> pd.DataFrame:
    """Removing data points whose URLs do not exist or are non-functional

    This is a heuristic way to filter out datapoints which correspond to websites without actual content.

    Args:
        df (pd.DataFrame): a generic pandas dataframe

    Returns:
        (pd.DataFrame) with errored out pages removed
    """
    bad_pages = []
    bad_pattern = r"Site Not Found|This page isn’t working|Internal Server Error|This page isn’t working|Your connection is not private|404"
    for i in range(len(df)):
        if re.search(bad_pattern, df['text'][i], flags=re.IGNORECASE):
            bad_pages.append(i)
    df = df.drop(labels=bad_pages, axis=0)
    df = df.reset_index(drop=True)
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Conversion of brands to numerical values via one hot encoding

    Args:
        df (pd.DataFrame): a pandas DataFrame that contains a 'brand_name' target label.

    Returns:
        (pd.DataFrame) with one hot encoded brand name labels
    """
    one_hot = pd.get_dummies(df.brand_name)
    df = pd.concat([df, one_hot], axis=1)
    return df


def get_top_level_domain(url: str) -> str:
    """Get the top level domain of a URL

    Args:
        url (str): a URL

    Returns:
        (str) the top level domain
    """
    tld = urlparse(url)
    return tld.netloc


def get_url_char_bigram_probs() -> Tuple[float]:
    """Get bigrams for each text(in our case it is n=2)

    Args:
         N/A

    Returns:
        Tuple(bg_cnts, ung_cnts): the probability of bigrams and unigrams occuring in the text
    """
    urls = pd.read_csv('links/top-1m.csv', names=['idx', 'URL'])
    urls = urls['URL']
    bg_cnts = {}
    ung_cnts = {}
    # going through each word in the corpus
    for url in urls:
        padding = ' '  # needed for bigram models
        url = padding + url.lower() + padding
        # getting characters
        chars = [c for c in url]
        bgs = nltk.bigrams(chars)  # getting bigrams

        # adding bigram to counts dictionary
        for bigram in bgs:
            bg_cnts[bigram] = bg_cnts[bigram] + 1 if bigram in bg_cnts else 1

        # adding unigram to count dictionary
        for unigram in chars:
            ung_cnts[
                unigram] = ung_cnts[unigram] + 1 if unigram in ung_cnts else 1

    # turning each count into a probability (MLE)
    for k, v in bg_cnts.items():
        first_char, _ = k
        denom = ung_cnts[first_char]
        bg_cnts[k] = v / denom  # now it is a conditional probability

    # same for unigrams
    norm = sum(ung_cnts.values())
    for k, v in ung_cnts.items():
        ung_cnts[k] = v / norm

    return bg_cnts, ung_cnts


def get_score(ung_probs: Dict[str, float], bg_probs: Dict[str, float],
              url_str: str) -> float:
    """Score of each url texts' bigrams

    Args:
        ung_probs (Dict[str, float]): unigram probabilities from the corpus
        bg_probs (Dict[str, float]): bigram probabilities from the corpus
        url_str (str): the url

    Returns:
        (float) a score of how likely the character sequence in the URL is
    """
    url_chars = [c for c in url_str]
    url_bgs = nltk.bigrams(url_chars)

    # getting a score of each url char bigram
    score = 0
    for bigram in url_bgs:
        # using linear interpolation smoothing (lambda 0.5)
        first_word, _ = bigram
        score += 0.5 * bg_probs.get(bigram, 0) + 0.5 * ung_probs.get(
            first_word, 0)
        # TODO probably should do entropy here
    return score


def split_data(df: pd.DataFrame, random_state=42) -> Tuple[np.array]:
    """
    Normal splitting of the dataset into training and testing

    Args:
        df (pd.DataFrame): a dataframe with a status columns
        random_state (int): the random state to ensure consistent splits

    Returns:
        (Tuple[np.array]) the split dataset
    """
    y = df['status']
    X = df.drop(['status'], axis=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X,
                                              y,
                                              test_size=0.1,
                                              random_state=random_state)
    return X_tr, X_te, y_tr, y_te


def store_datasets(x_train: np.array, x_test: np.array, y_train: np.array,
                   y_test: np.array) -> None:
    """Save the CSVs of the split data to the data directory

    Args:
        x_train (np.array): training split of X labels
        x_test (np.array): test split of X labels
        y_train (np.array): training split of y labels
        y_test (np.array): testing split of y labels

    Returns:
        None, saves the values to data/ directory
    """
    x_train.to_csv('data/X_train.csv', index=False)
    x_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)


if __name__ == "__main__":
    data = pd.read_csv("scraped/all_data_scraped.csv")
    data = drop_nulls_and_duplicates(data)
    data = drop_non_unique_text(data)
    data = drop_bad_pages(data)
    data = one_hot_encode(data)
    data = data.drop(['brand_name', 'no_brand'], axis=1)

    # URL based features
    bg_probs, ung_probs = get_url_char_bigram_probs()
    tld_lens = []
    url_scores = []
    for url in data['URL']:
        tld = get_top_level_domain(url)
        tld_len = len(tld)  # the len of the tld is a feature
        score = get_score(ung_probs, bg_probs, tld)
        tld_lens.append(tld_len)
        url_scores.append(score)

    # text based features
    text_lens = []
    for t in data['text']:
        text_lens.append(len(t))

    # adding new features to data
    data['url_len'] = tld_lens
    data['tld_char_score'] = url_scores
    data['text_len'] = text_lens
    # dropping all text features
    data = data.drop(["URL", "text"], axis=1)

    X_train, X_test, y_train, y_test = split_data(data)
    store_datasets(X_train, X_test, y_train, y_test)
