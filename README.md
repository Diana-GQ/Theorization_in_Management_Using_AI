# Theorization in Management using AI
Applying LLMs to select relevant data for qualitative analysis in management research.
This repository contains the code for the paper "Enhancing Theorization Using Artificial Inteligence: Leveraging Large Language Models for Qualitative Analysis of Online Data" by Garcia Quevedo _et al._

## Introduction

The project shows how to use NLP algorithms and Large Language Models (LLM) to analyze social media posts using the following scripts:

**preprocess_posts.py**:

Loads a collection of .csv files containing posts, cleans up the text, tokenizes words, adds attributes to each post, and saves to a parquet file.

How to run: 

``` python3 preprocess_posts.py --input_data_path <path_to_collection_of_csv_files> --output_file_name <name_of_output_file> ```

Example of input data path: `data/**/dataset_scraper*.csv` (will retrieve the list of files that match the expression).

Example of output file name: `pre_processed_posts.parquet`.



**add_sentiment.py**:

Uses a sentiment analysis model to classify each post as `positive`, `negative` or `neutral`.

How to run: 

``` python3 add_sentiment.py --input_file_name <parquet_file_with_pre_processed_posts> --output_file_name <name_of_output_file> ```

Example of input file name: `pre_processed_posts.parquet` (produced by the `preprocess_posts.py` script).

Example of output file name: `posts_with_sentiment.parquet`.


**add_ada_text_embeddings.py**:

Generates a text embedding for each post's text using the OpenAI's ADA embedding model (`text-embedding-ada-002`).

How to run: 

``` python3 add_ada_text_embeddings.py --input_file_name <parquet_file_with_pre_processed_posts> --output_file_name <name_of_output_file> ```

Example of input file name: `posts_with_sentiment.parquet` (produced by the `add_sentiment.py` script).

Example of output file name: `posts_with_ada_embeddings.parquet`.


**add_gpt_topics.py**:

Generates topics from the collection of posts and assign the relevant topics to each post using an OpenAI GPT model (e.g., `gpt-4o`).

How to run: 

``` python3 add_gpt_topics.py --input_file_name <parquet_file_with_pre_processed_posts> --output_file_name <name_of_output_file> ```

Example of input file name: `posts_with_ada_embeddings.parquet` (produced by the `add_ada_text_embeddings.py` script).

Example of output file name: `posts_with_gpt_topics.parquet`.

