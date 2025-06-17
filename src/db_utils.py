import os

from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Get MongoDB connection string from environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")


def get_mongodb_client():
    """Returns a MongoDB client instance"""
    return MongoClient(MONGODB_URI)


def get_database(database_name="data_viz_ai"):
    """Returns a MongoDB database instance"""
    client = get_mongodb_client()
    return client[database_name]


def store_dataset(dataset_name, dataset_df):
    """Stores a pandas DataFrame in MongoDB"""
    db = get_database()
    collection = db[dataset_name]

    # Convert DataFrame to list of dictionaries and insert into MongoDB
    records = dataset_df.to_dict("records")

    # Delete existing records if any
    collection.delete_many({})

    # Insert new records
    result = collection.insert_many(records)
    return len(result.inserted_ids)


def get_dataset_names():
    """Returns a list of dataset names stored in MongoDB"""
    db = get_database()
    return db.list_collection_names()


def get_dataset(dataset_name):
    """Returns a dataset from MongoDB as a pandas DataFrame"""
    import pandas as pd

    db = get_database()
    collection = db[dataset_name]

    # Get all documents from the collection
    cursor = collection.find({}, {"_id": 0})

    # Convert to DataFrame
    df = pd.DataFrame(list(cursor))
    return df
