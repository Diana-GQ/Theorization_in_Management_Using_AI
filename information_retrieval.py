import argparse
import logging
import os

import openai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from constants import *
from openai import OpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]

logging.basicConfig(level=logging.DEBUG, 
                    format='[%(levelname)s] %(asctime)s - %(message)s')


def get_embedding(text: str, client: OpenAI, model: str = ADA_EMBEDDING_MODEL) -> list:
    """
    Generates an embedding for the given text using the specified model.

    Args:
        text (str): The text to generate an embedding for.
        client (OpenAI): The OpenAI client to use for generating the embedding.
        model (str): The model to use for generating the embedding.

    Returns:
        list: The generated embedding.
    """
    text = text.replace('\n', ' ')
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def retrieve_similar_entries(df: pd.DataFrame, query_embedding: list, top_n: int) -> pd.DataFrame:
    """
    Retrieves the top N entries in the DataFrame with the highest cosine similarity to the query embedding.

    Args:
        df (pd.DataFrame): DataFrame containing posts with 'ada_embedding' column.
        query_embedding (list): Embedding of the query string.
        top_n (int): Number of top entries to retrieve.

    Returns:
        pd.DataFrame: DataFrame with the top N similar entries.
    """
    df['similarity'] = df['ada_embedding'].apply(lambda x: cosine_similarity([x], [query_embedding])[0][0])
    return df.nlargest(top_n, 'similarity')


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
        "--query",
        default=None,
        type=str,
        required=True,
        help="Query string to find similar posts."
    )
    parser.add_argument(
        "--top_n",
        default=5,
        type=int,
        required=False,
        help="Number of top similar posts to retrieve."
    )
    args = parser.parse_args()

    df_posts = pd.read_parquet(args.input_file_name)
    client = OpenAI()
    query_embedding = get_embedding(args.query, client)

    top_similar_posts = retrieve_similar_entries(df_posts, query_embedding, args.top_n)
    print(top_similar_posts)


if __name__ == "__main__":
    main()
