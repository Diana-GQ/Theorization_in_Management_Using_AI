import argparse
import logging
import os
import openai
import pandas as pd
 
from constants import *
from openai import OpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]

logging.basicConfig(level=logging.DEBUG, 
                    format='[%(levelname)s] %(asctime)s - %(message)s')


def get_embedding(text, client, model=ADA_EMBEDDING_MODEL):
   text = text.replace('\n', ' ')
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def add_ada_embeddings(df_posts):
    client = OpenAI()

    df_posts['ada_embedding'] = df_posts['text'].apply(lambda x: get_embedding(x, client))
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
        help="Name of parquet file with ada embeddings added to posts."
    )
    args = parser.parse_args()

    df_posts = pd.read_parquet(args.input_file_name)
    df_posts = add_ada_embeddings(df_posts)

    df_posts.to_parquet(args.output_file_name, index=False)


if __name__ == "__main__":
    main()
