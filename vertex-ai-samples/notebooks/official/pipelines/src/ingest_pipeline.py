# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# General imports
from __future__ import absolute_import
import argparse
import logging
import os
import string

# Preprocessing imports
import tensorflow as tf
import bs4
import nltk

import apache_beam as beam
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


# Helpers ---------------------------------------------------------------------

def get_args():
    """
    Get command line arguments.
    Returns:
      args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', dest='inputs', default='data/raw/reuters/*.sgm',
                        help='A directory location of input data')
    parser.add_argument('--bq-dataset', dest='bq_dataset', required=False,
                        default='reuters_dataset', help='Dataset name used in BigQuery.')
    parser.add_argument('--bq-table', dest='bq_table', required=False,
                        default='reuters_ingested_table', help='Table name used in BigQuery.')
    args, pipeline_args = parser.parse_known_args()
    return args, pipeline_args

def get_paths(data_pattern):
    """
  A function to get all the paths of the files in the data directory.
  Args:
    data_pattern: A directory location of input data.
  Returns:
    A list of file paths.
  """
    data_paths = tf.io.gfile.glob(data_pattern)
    return data_paths


def get_title(article):
    """
    A function to get the title of an article.
    Args:
        article: A BeautifulSoup object of an article.
    Returns:
        A string of the title of the article.
    """
    title = article.find('text').title
    if title is not None:
        title = ''.join(filter(lambda x: x in set(string.printable), title.text))
        title = title.encode('ascii', 'ignore')
    return title


def get_content(article):
    """
    A function to get the content of an article.
    Args:
        article: A BeautifulSoup object of an article.
    Returns:
        A string of the content of the article.
    """
    content = article.find('text').body
    if content is not None:
        content = ''.join(filter(lambda x: x in set(string.printable), content.text))
        content = ' '.join(content.split())
        try:
            content = '\n'.join(nltk.sent_tokenize(content))
        except LookupError:
            nltk.download('punkt')
            content = '\n'.join(nltk.sent_tokenize(content))
        content = content.encode('ascii', 'ignore')
    return content


def get_topics(article):
    """
    A function to get the topics of an article.
    Args:
        article: A BeautifulSoup object of an article.
    Returns:
        A list of strings of the topics of the article.
    """
    topics = []
    for topic in article.topics.children:
        topic = ''.join(filter(lambda x: x in set(string.printable), topic.text))
        topics.append(topic.encode('ascii', 'ignore'))
    return topics


def get_articles(data_paths):
    """
    Args:
        data_paths: A list of file paths.
    Returns:
        A list of articles.
    """
    data = tf.io.gfile.GFile(data_paths, 'rb').read()
    soup = bs4.BeautifulSoup(data, "html.parser")
    articles = []
    for raw_article in soup.find_all('reuters'):
        article = {
            'title': get_title(raw_article),
            'content': get_content(raw_article),
            'topics': get_topics(raw_article)
        }
        if None not in article.values():
            if [] not in article.values():
                articles.append(article)
    return articles


def get_bigquery_schema():
    """
    A function to get the BigQuery schema.
    Returns:
        A list of BigQuery schema.
    """

    table_schema = bigquery.TableSchema()
    columns = (('topics', 'string', 'repeated'),
               ('title', 'string', 'nullable'),
               ('content', 'string', 'nullable'))

    for column in columns:
        column_schema = bigquery.TableFieldSchema()
        column_schema.name = column[0]
        column_schema.type = column[1]
        column_schema.mode = column[2]
        table_schema.fields.append(column_schema)

    return table_schema


# Pipeline runner
def run(args, pipeline_args=None):
    """
    A function to run the pipeline.
    Args:
        args: The parsed arguments.
    Returns:
        None
    """

    options = PipelineOptions(pipeline_args)
    options.view_as(SetupOptions).save_main_session = True

    pipeline = beam.Pipeline(options=options)
    articles = (
            pipeline
            | 'Get Paths' >> beam.Create(get_paths(args.inputs))
            | 'Get Articles' >> beam.Map(get_articles)
            | 'Get Article' >> beam.FlatMap(lambda x: x)
    )
    if options.get_all_options()['runner'] == 'DirectRunner':
        articles | 'Dry run' >> beam.io.WriteToText('data/processed/reuters', file_name_suffix=".jsonl")
    else:
        (articles
         | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                    project=options.get_all_options()['project'],
                    dataset=args.bq_dataset,
                    table=args.bq_table,
                    schema=get_bigquery_schema(),
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE)
         )
    job = pipeline.run()

    if options.get_all_options()['runner'] == 'DirectRunner':
        job.wait_until_finish()


if __name__ == '__main__':
    args, pipeline_args = get_args()
    logging.getLogger().setLevel(logging.INFO)
    run(args, pipeline_args)
