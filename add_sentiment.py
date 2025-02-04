import argparse
import logging
import pandas as pd

from transformers import pipeline

logging.basicConfig(level=logging.DEBUG, 
                    format='[%(levelname)s] %(asctime)s - %(message)s')


# From https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
# twitter-XLM-roBERTa-base for Sentiment Analysis
# This is a multilingual XLM-roBERTa-base model trained on ~198M tweets and
# finetuned for sentiment analysis. The sentiment fine-tuning was done on 8
# languages (Ar, En, Fr, De, Hi, It, Sp, Pt) but it can be used for more
# languages (see paper for details).
#
# Paper: XLM-T: A Multilingual Language Model Toolkit for Twitter
# https://arxiv.org/pdf/2104.12250.pdf
def add_sentiment(df_posts):
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sentiment_task = pipeline(
        "sentiment-analysis",
        model=model_path,
        tokenizer=model_path,
        # Set max_length since FB posts can be longer than tweets
        truncation=True, max_length=512, add_special_tokens = True
    )
    sa = sentiment_task([str(x) for x in df_posts['text']])
    df_posts['sentiment'] = [x['label'] for x in sa]
    df_posts['sentiment_score'] = [x['score'] for x in sa]

    return df_posts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file_name",
        default=None,
        type=str,
        required=True,
        help="Name of parquet file with processed posts."
    )
    parser.add_argument(
        "--output_file_name",
        default=None,
        type=str,
        required=True,
        help="Name of parquet file with sentiment added to posts."
    )
    args = parser.parse_args()

    df_posts = pd.read_parquet(args.input_file_name)
    df_posts = add_sentiment(df_posts)

    df_posts.to_parquet(args.output_file_name, index=False)


if __name__ == "__main__":
    main()
