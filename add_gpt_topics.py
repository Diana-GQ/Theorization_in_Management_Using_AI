import argparse
import json
import logging
import os
import random

import openai
import pandas as pd
import tiktoken
from pydantic import BaseModel
from tqdm.auto import tqdm

from constants import *
from openai import OpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI()

logging.basicConfig(level=logging.DEBUG, 
                    format='[%(levelname)s] %(asctime)s - %(message)s')


class Topic(BaseModel):
    name: str
    description: str

class TopicList(BaseModel):
    topics: list[Topic]


def get_prompt(docs: list[str]) -> list[dict]:
    """
    Generates a prompt for the GPT model to identify topics in social media posts.

    Args:
        docs (list[str]): List of social media posts.

    Returns:
        list[dict]: List of messages for the GPT model.
    """
    delimiter = '###'
    system_message = '''
        You're a helpful assistant. Your task is to analyse social media posts.
    '''
    user_message = f'''
        Below is a representative set of posts delimited with {delimiter}. 
        Please identify the ten most mentioned topics in these comments.
        The topics must be mutually exclusive.
        A concise description must be provided for each topic.
        The results must be in English.

        Social media posts:
        {delimiter}
        {delimiter.join(docs)}
        {delimiter}
    '''
    messages =  [  
        {'role':'system', 
         'content': system_message},    
        {'role':'user', 
         'content': f"{user_message}"},  
    ]
    return messages


def generate_sublists(input_list: list[int], limit: int) -> list[list[int]]:
    """
    Generates sublists of indices where the sum of lengths does not exceed the limit.

    Args:
        input_list (list[int]): List of lengths of social media posts.
        limit (int): Token limit for the GPT model.

    Returns:
        list[list[int]]: List of sublists of indices.
    """
    result = []
    current_sublist = []
    current_sum = 0
    
    random.shuffle(input_list)
    for idx, num in enumerate(input_list):
        if current_sum + num > limit:
            result.append(current_sublist)
            current_sublist = [idx]
            current_sum = num
        else:
            current_sublist.append(idx)
            current_sum += num
    
    if current_sublist:
        result.append(current_sublist)
    
    return result


def reduce_topics(df: pd.DataFrame, model: str = GPT_MODEL) -> pd.DataFrame:
    """
    Reduces the list of topics to up to ten by removing duplicates.

    Args:
        df (pd.DataFrame): DataFrame containing topics and their descriptions.
        model (str): The GPT model to use for reducing topics.

    Returns:
        pd.DataFrame: DataFrame with reduced topics.
    """
    system_message = '''
        You're a helpful assistant. Your task is to analyse social media posts.
    '''
    user_message = f'''
        Below is a set of topics and their descriptions. 
        Reduce the list to up to ten topics by removing duplicated topics.

        Topics:
        {df.to_json(orient='records')}
    '''
    messages =  [  
        {'role':'system', 
        'content': system_message},    
        {'role':'user', 
        'content': f"{user_message}"},  
    ]
    result = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            'type': 'json_schema',
            'json_schema': 
                {
                    "name":"whocares", 
                    "schema": TopicList.schema()
                }
            },
    )
    topics = json.loads(result.choices[0].message.content).get('topics', [])
    df = pd.DataFrame(topics)

    return df


def generate_gpt_topics(df: pd.DataFrame, model: str = GPT_MODEL, token_limit: int = GPT_TOKEN_LIMIT) -> pd.DataFrame:
    """
    Generates GPT topics for a DataFrame of posts.

    Args:
        df (pd.DataFrame): DataFrame containing posts with a 'text' column.
        model (str): The GPT model to use for generating topics.
        token_limit (int): Token limit for the GPT model.

    Returns:
        pd.DataFrame: DataFrame with generated topics.
    """
    gpt_enc = tiktoken.encoding_for_model(GPT_MODEL_ENCODING)
    docs = df.text
    lengths = [len(gpt_enc.encode(x)) for x in docs]
    sublists = generate_sublists(lengths, token_limit)

    df_topics_all = pd.DataFrame()
    for sl in sublists:
        docs_sl = [docs.values[i] for i in sl]
        messages = get_prompt(docs_sl)
        result = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                'type': 'json_schema',
                'json_schema': 
                    {
                        "name":"whocares", 
                        "schema": TopicList.schema()
                    }
                },
        )
        topics = json.loads(result.choices[0].message.content).get('topics', [])
        df_topics = pd.DataFrame(topics)
        df_topics_all = pd.concat([df_topics_all, df_topics])

    df_topics_all = reduce_topics(df_topics_all)

    return df_topics_all


def get_prompt_topic_mapping(doc: str, topic_list: str) -> list[dict]:
    """
    Generates a prompt for the GPT model to map topics to a social media post.

    Args:
        doc (str): Social media post.
        topic_list (str): List of topics.

    Returns:
        list[dict]: List of messages for the GPT model.
    """
    delimiter = '###'
    system_message = '''
        You're a helpful assistant. Your task is to analyse social media posts.
    '''
    user_message = f'''
        Below is a social media post delimited with {delimiter}. 
        Please, identify the main topics mentioned in this post from the list of topics below. 

        Output is a list with the following format
        <topic1>, <topic2>, ...

        Include only topics from the provided below list.
        If none of the topics from the list is identified, return the word "None".

        List of topics:
        {topic_list}

        Social media post:
        {delimiter}
        {doc}
        {delimiter}
    '''
    messages =  [  
        {'role':'system', 
         'content': system_message},    
        {'role':'user', 
         'content': f"{user_message}"},  
    ]

    return messages


def get_model_response(messages: list[dict], model: str = GPT_MODEL) -> str:
    """
    Gets the response from the GPT model for the given messages.

    Args:
        messages (list[dict]): List of messages for the GPT model.
        model (str): The GPT model to use.

    Returns:
        str: The response from the GPT model.
    """
    result = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return result.choices[0].message.content


def assign_topics(df_posts: pd.DataFrame, df_topics: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns topics to a DataFrame of posts.

    Args:
        df_posts (pd.DataFrame): DataFrame containing posts with a 'text' column.
        df_topics (pd.DataFrame): DataFrame containing topics.

    Returns:
        pd.DataFrame: DataFrame with assigned topics.
    """
    topic_list = '\n'.join(df_topics.name)
    docs = df_posts.text
    for doc in tqdm(docs):
        messages = get_prompt_topic_mapping(doc, topic_list)
        topics = get_model_response(messages)
        topics = [f'gpt_topic: {t.lstrip()}' for t in topics.split(',')]
        for t in topics:
            df_posts.loc[df_posts['text'] == doc, t] = 1
    df_posts.fillna(0, inplace=True)

    return df_posts


def add_gpt_topics(df_posts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds GPT topics to a DataFrame of posts.

    Args:
        df_posts (pd.DataFrame): DataFrame containing posts with a 'text' column.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrame with added GPT topics and DataFrame with the list of GPT topics.
    """
    df_topics = generate_gpt_topics(df_posts)
    df_posts = assign_topics(df_posts, df_topics)

    return df_posts, df_topics


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
        help="Name of parquet file with GPT topics added to posts."
    )
    parser.add_argument(
        "--topics_file_name",
        default="gpt_topics.parquet",
        type=str,
        required=False,
        help="Name of output parquet file with the list of GPT topics."
    )
    args = parser.parse_args()

    df_posts = pd.read_parquet(args.input_file_name)
    df_posts, df_topics = add_gpt_topics(df_posts)

    df_posts.to_parquet(args.output_file_name, index=False)
    df_topics.to_parquet(args.topics_file_name, index=False)


if __name__ == "__main__":
    main()
