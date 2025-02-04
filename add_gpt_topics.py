import argparse
import json
import logging
import os
import openai
import pandas as pd
import random
import tiktoken

from constants import *
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI()

logging.basicConfig(level=logging.DEBUG, 
                    format='[%(levelname)s] %(asctime)s - %(message)s')


class Topic(BaseModel):
    name: str
    description: str

class TopicList(BaseModel):
    topics: list[Topic]


def get_prompt(docs):
    delimiter = '###'
    system_message = '''
        You're a helpful assistant. Your task is to analyse social media posts.
    '''
    user_message = f'''
        Below is a representative set of posts delimited with {delimiter}. 
        Please identify the ten most mentioned topics in these comments.
        The topics must be mutually exclusive.
        A concise description mist be provided for each topic.
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


def generate_sublists(input_list, limit):
    # Initialize an empty list to hold the result
    result = []
    # Initialize an empty sublist to accumulate indices
    current_sublist = []
    current_sum = 0  # This will keep track of the current sum
    
    random.shuffle(input_list)
    for idx, num in enumerate(input_list):
        # If adding the current number exceeds the limit
        if current_sum + num > limit:
            # Append the current sublist to the result
            result.append(current_sublist)
            # Start a new sublist with the current index
            current_sublist = [idx]
            current_sum = num  # Reset the sum to the current number
        else:
            # Otherwise, add the current index to the sublist
            current_sublist.append(idx)
            current_sum += num  # Add the number to the current sum
    
    # Append the last sublist to the result (if not empty)
    if current_sublist:
        result.append(current_sublist)
    
    return result


def reduce_topics(df, model=GPT_MODEL):
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


def generate_gpt_topics(df, model=GPT_MODEL, token_limit=GPT_TOKEN_LIMIT):
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


def get_prompt_topic_mapping(doc, topic_list):
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


def get_model_response(messages, model=GPT_MODEL):
    result = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return result.choices[0].message.content


def assign_topics(df_posts, df_topics):
    topic_list = '\n'.join(df_topics.name)
    docs = df_posts.text
    for doc in tqdm(docs):
        messages = get_prompt_topic_mapping(doc, topic_list)
        topics = get_model_response(messages)
        topics = [f'gpt_topic: {t.lstrip()}' for t in topics.split(',')]
        for t in topics:
            df_posts.loc[df_posts['text']==doc,t] = 1
    df_posts.fillna(0, inplace=True)

    return df_posts


def add_gpt_topics(df_posts):
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
