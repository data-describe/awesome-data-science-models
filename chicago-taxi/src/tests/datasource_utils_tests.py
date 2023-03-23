# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test utilities for generating BigQuery data querying scirpts."""

import sys
import os
import logging
from google.cloud import bigquery

from src.common import datasource_utils

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

LIMIT = 100

TARGET_COLUMN = "tip_bin"

EXPECTED_TRAINING_COLUMNS = [
    "trip_month",
    "trip_day",
    "trip_day_of_week",
    "trip_hour",
    "trip_seconds",
    "trip_miles",
    "payment_type",
    "pickup_grid",
    "dropoff_grid",
    "euclidean",
    "loc_cross",
    "tip_bin",
]


def test_training_query():

    project = os.getenv("PROJECT")
    location = os.getenv("BQ_LOCATION")
    bq_dataset_name = os.getenv("BQ_DATASET_NAME")
    bq_table_name = os.getenv("BQ_TABLE_NAME")

    assert project, "Environment variable PROJECT is None!"
    assert location, "Environment variable BQ_LOCATION is None!"
    assert bq_dataset_name, "Environment variable BQ_DATASET_NAME is None!"
    assert bq_table_name, "Environment variable BQ_TABLE_NAME is None!"

    logging.info(f"BigQuery Source: {project}.{bq_dataset_name}.{bq_table_name}")

    query = datasource_utils._get_source_query(
        bq_dataset_name=bq_dataset_name,
        bq_table_name=bq_table_name,
        ml_use="UNASSIGNED",
        limit=LIMIT,
    )

    bq_client = bigquery.Client(project=project, location=location)
    df = bq_client.query(query).to_dataframe()
    columns = set(df.columns)
    assert columns == set(EXPECTED_TRAINING_COLUMNS)
    assert df.shape == (LIMIT, 12)


def test_serving_query():

    project = os.getenv("PROJECT")
    location = os.getenv("BQ_LOCATION")
    bq_dataset_name = os.getenv("BQ_DATASET_NAME")
    bq_table_name = os.getenv("BQ_TABLE_NAME")

    assert project, "Environment variable PROJECT is None!"
    assert location, "Environment variable BQ_LOCATION is None!"
    assert bq_dataset_name, "Environment variable BQ_DATASET_NAME is None!"
    assert bq_table_name, "Environment variable BQ_TABLE_NAME is None!"

    logging.info(f"BigQuery Source: {project}.{bq_dataset_name}.{bq_table_name}")

    query = datasource_utils._get_source_query(
        bq_dataset_name=bq_dataset_name,
        bq_table_name=bq_table_name,
        ml_use=None,
        limit=LIMIT,
    )

    bq_client = bigquery.Client(project=project, location=location)
    df = bq_client.query(query).to_dataframe()
    columns = set(df.columns)
    expected_serving_columns = EXPECTED_TRAINING_COLUMNS
    expected_serving_columns.remove(TARGET_COLUMN)
    assert columns == set(expected_serving_columns)
    assert df.shape == (LIMIT, 11)
