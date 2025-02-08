import argparse
import glob
import logging
import re
import string
from typing import List

import pandas as pd
import spacy
from nltk.tokenize import TweetTokenizer
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

from constants import *

logging.basicConfig(level=logging.DEBUG, 
                    format='[%(levelname)s] %(asctime%s - %(message)s')


punctuation = set(string.punctuation)
stopwords = list(fr_stop) + list(en_stop)
# Words to remove, on top of punctuation, stop words, etc.
misc_blocked = set([
    'e', 'h', 'q', 'nan', u'-', u'-ce', u'-elle', u'-en', u'-il', u'-là',
    u'-moi', u'-vou', u'-vous', u'-t', u'être', u'avoir', u'aussi', u'cela',
    u'faire', u'merci', u'l', u'd', u'c', u's', u'j', u'n', u'm', u'qu', u'thi',
    u'haver', u't'
])
# Instantiate tokenizer class
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                           reduce_len=True)
# Initialize spacy for lemmatization of French text
# https://stackoverflow.com/questions/13131139/lemmatize-french-text
nlp = spacy.load('fr_core_news_md')

emoj = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff"
    u"\u2600-\u2B55"
    u"\u200d"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\ufe0f"
    u"\u3030"
    u"\u200c"
                    "]+", re.UNICODE)


def remove_emojis(data: str) -> str:
    """
    Remove emojis from the input text.

    Arguments:
        data (str): The input text from which emojis need to be removed.

    Returns:
        str: The text with emojis removed.
    """
    return re.sub(emoj, '', data)


def count_emojis(data: str) -> int:
    """
    Count the number of emojis in the input text.

    Arguments:
        data (str): The input text from which emojis need to be counted.

    Returns:
        int: The number of emojis in the text.
    """
    return sum([len(n) for n in re.findall(emoj, data)])


def normalize_text(x: str) -> str:
    """
    Normalize the input text by removing hyperlinks, emails, phone numbers, and other unwanted characters.

    Arguments:
        x (str): The input text to be normalized.

    Returns:
        str: The normalized text.
    """
    # Remove hyperlinks, emails, phone numbers
    regex_str = (r'https?://[^\s\n\r]+|www.[^\s\n\r]+|bit.ly[^\s\n\r]+|'
                 r'[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}|'
                 r'\d{2}.\d{2}.\d{2}.\d{2}.\d{2}|'
                 r'\d+|…|\u200d|»|«|“|•|°|->|…')
    pattern = re.compile(regex_str)
    x = pattern.sub('', x)
    # Replace other characters with spaces before tokenization
    pattern = re.compile(r"\n|\xa0|'|’|@|\.|#\w+\s*")
    x = pattern.sub(' ', x)
    return x


def cleanup(x: str) -> List:
    """ Clean up, tokenize and lemmatize an input string.

    Arguments:
        x (str): The text to clean-up, tokenize and lemmatize.
    Return:
        list[(str, str)]: List of pairs. Each pair is a token from the original
                          text: <not-lemmatized, lemmatized>
    """
    x = normalize_text(x)
    x = remove_emojis(x)

    # Lemmatize tokens and remove stop words
    tokens = [
        (t.text, t.lemma_) for t in nlp(' '.join(tokenizer.tokenize(x)))
        if t.lemma_ not in stopwords and
           t.lemma_ not in punctuation and
           t.lemma_ not in misc_blocked
    ]
    return tokens


def load_posts(input_data_path: str) -> pd.DataFrame:
    """
    Load posts from CSV files, concatenate them into a single DataFrame, and remove duplicates.

    Arguments:
        input_data_path (str): Path to the directory containing the CSV files.

    Returns:
        pd.DataFrame: DataFrame containing the loaded and concatenated posts.
    """
    csv_profiles = glob.glob(input_data_path, recursive=True)
    acc = []
    for f in csv_profiles:
        source = f.split('/')[-2]  # Name of the dir containing the file
        df_tmp = pd.read_csv(
            f,
            usecols=twitter_columns,
            parse_dates=['created_at']
        )
        df_tmp.rename(columns={'full_text': 'text'}, inplace=True)
        df_tmp['author'] = source
        acc.append(df_tmp)
    df = pd.concat(acc)
    # Remove duplicates
    old_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    msg = f'Removed {old_rows - len(df)} duplicates from {old_rows} rows.'
    logging.info(msg)

    return df


def add_tokens_and_lemmas(df_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Add tokenized and lemmatized text columns to the DataFrame.

    Arguments:
        df_posts (pd.DataFrame): DataFrame containing the posts (as 'text' column).

    Returns:
        pd.DataFrame: DataFrame with additional 'tokens' and 'lemmas' columns.
    """
    tokens = df_posts['text'].map(cleanup)
    df_posts['tokens'] = tokens.map(lambda x: ' '.join([w[0] for w in x]))
    df_posts['lemmas'] = tokens.map(lambda x: ' '.join([w[1] for w in x]))

    return df_posts


def add_author_attributes(df_posts: pd.DataFrame, attributes_file: str) -> pd.DataFrame:
    """
    Add author attributes to the DataFrame by merging with an attributes file.

    Arguments:
        df_posts (pd.DataFrame): DataFrame containing the posts.
        attributes_file (str): Path to the CSV file containing author attributes.

    Returns:
        pd.DataFrame: DataFrame with additional author attributes.
    """
    df_attr = pd.read_csv(attributes_file)
    df_posts = pd.merge(df_posts, df_attr, how='left', on='author')

    return df_posts


def add_emoji_count(df_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column to the DataFrame that counts the number of emojis in each post.

    Arguments:
        df_posts (pd.DataFrame): DataFrame containing the posts (as 'text' column).

    Returns:
        pd.DataFrame: DataFrame with an additional 'emoji_count' column.
    """
    df_posts['emoji_count'] = df_posts['text'].map(count_emojis)

    return df_posts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data_path",
        default=None,
        type=str,
        required=True,
        help="Path to scrapped posts."
    )
    parser.add_argument(
        "--output_file_name",
        default=None,
        type=str,
        required=True,
        help="Name of parquet file with processed posts."
    )
    args = parser.parse_args()

    df_posts = load_posts(args.input_data_path)
    df_posts = add_tokens_and_lemmas(df_posts)
    df_posts = add_author_attributes(df_posts, ATTRIBUTES_FILE)
    df_posts = add_emoji_count(df_posts)

    df_posts.to_parquet(args.output_file_name, index=False)


if __name__ == "__main__":
    main()
