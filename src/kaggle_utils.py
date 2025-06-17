import os

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def authenticate_kaggle():
    """Authenticates with Kaggle API"""
    api = KaggleApi()
    api.authenticate()
    return api


def search_datasets(query, max_results=10):
    """Searches for datasets on Kaggle"""
    api = authenticate_kaggle()
    datasets = api.dataset_list(search=query, max_size=None, license_name=None)

    # Return only the first max_results datasets
    return datasets[:max_results]


def download_dataset(dataset_ref, path="./data"):
    """Downloads a dataset from Kaggle"""
    api = authenticate_kaggle()

    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download the dataset
    api.dataset_download_files(dataset_ref, path=path, unzip=True)

    # Get a list of CSV files in the directory
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

    if not csv_files:
        return None, "No CSV files found in the dataset"

    # Load the first CSV file
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    return df, csv_files[0]
