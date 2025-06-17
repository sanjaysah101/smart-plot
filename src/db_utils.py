import pandas as pd  # Added import for DataFrame manipulation
import streamlit as st
from pymongo import MongoClient

from src.ai_utils import generate_text_embedding  # Added import

# Get MongoDB connection string from environment variables
MONGODB_URI = st.secrets["MONGODB_URI"]


def get_mongodb_client():
    """Returns a MongoDB client instance"""
    return MongoClient(MONGODB_URI)


def get_database(database_name="data_viz_ai"):
    """Returns a MongoDB database instance"""
    client = get_mongodb_client()
    return client[database_name]


def store_dataset(dataset_name, dataset_df, text_column_for_embedding=None):
    """Stores a pandas DataFrame in MongoDB, generates embeddings, and creates a vector index."""
    db = get_database()
    collection = db[dataset_name]

    # Generate embeddings if a text column is specified
    if text_column_for_embedding and text_column_for_embedding in dataset_df.columns:
        # Ensure the text column is of string type, fill NaNs with empty strings
        dataset_df[text_column_for_embedding] = (
            dataset_df[text_column_for_embedding].astype(str).fillna("")
        )
        # Generate embeddings in batches or one by one
        # Note: Batching would be more efficient for large datasets
        dataset_df["embedding"] = dataset_df[text_column_for_embedding].apply(
            lambda x: generate_text_embedding(x) if x else [0.0] * 768
        )
        # Or keep it if it's a primary column users might want to see
    # Drop the original text column if it's a concatenation or temporary
    else:
        # If no specific column, or column doesn't exist, store without embeddings
        # Or, alternatively, try to concatenate all string columns (more complex)
        print(
            f"Warning: Text column '{text_column_for_embedding}' not found or not specified. Storing data without embeddings."
        )

    records = dataset_df.to_dict("records")

    collection.delete_many({})
    result = collection.insert_many(records)

    # Create vector index if embeddings were generated
    if "embedding" in dataset_df.columns:
        try:
            create_vector_index(
                collection, "embedding", 768
            )  # 768 is dimension for text-embedding-004
            print(
                f"INFO: Embeddings generated for collection '{dataset_name}'. "
                f"Please ensure a vector search index (e.g., named 'vector_index') is manually created in MongoDB Atlas "
                f"on the 'embedding' field using the definition provided by 'create_vector_index' logs."
            )
        except Exception as e:
            print(f"Error during vector index guidance for '{dataset_name}': {e}")

    return len(result.inserted_ids)


def create_vector_index(
    collection, field_name, vector_dimension, index_name="vector_index"
):
    """Creates a vector search index on the specified field."""
    # Check if the index already exists
    # Note: PyMongo's list_indexes() might not show search indexes directly.
    # Managing search indexes is often done via Atlas UI or Atlas Admin API.
    # For self-managed MongoDB, vector search capabilities depend on version and configuration.
    # This is a simplified representation. Actual index creation for Atlas Search:
    try:
        # Example for Atlas Search (requires Atlas Admin API or UI setup for search indexes)
        # This command is illustrative; actual creation is via Atlas UI or specific API calls
        # For on-prem/community, vector search might use different mechanisms or not be available directly like this.

        # A more direct way if using a version of MongoDB that supports $vectorSearch natively
        # and allows index creation via commands (e.g., MongoDB 6.0+ with specific index types like HNSW)
        # db.command({ createIndexes: collection.name, indexes: [{ key: { [field_name]: "vector" }, name: index_name, vectorOptions: { type: "hnsw", dimensions: vector_dimension, similarity: "cosine" }}]})
        # The above is a conceptual command structure. Actual syntax may vary.

        # For Atlas, search indexes are defined separately.
        # This function assumes the index needs to be created if not present.
        # We'll define a search index structure. This is typically done once.
        search_index_model = {
            "name": index_name,
            "definition": {
                "mappings": {
                    "dynamic": True,  # or False, depending on your needs
                    "fields": {
                        field_name: [
                            {
                                "type": "vector",
                                "dimensions": vector_dimension,
                                "similarity": "cosine",  # or "euclidean", "dotProduct"
                            }
                        ]
                    },
                }
            },
        }

        # This is a conceptual representation. Actual index creation on Atlas
        # is usually done through the Atlas UI, Atlas CLI, or Atlas Admin API.
        # PyMongo itself doesn't have a direct `create_search_index` method like this for Atlas Search.
        # We'll print a message indicating the need for manual setup or Atlas API usage.
        print(
            f"INFO: Vector search index '{index_name}' on field '{field_name}' needs to be created in MongoDB Atlas."
        )
        print(f"Index definition to use: {search_index_model}")
        # Example: collection.create_index([(field_name, "text")]) # This is for text index, not vector
        # If your MongoDB server and version support it directly via a command:
        # try:
        #     collection.database.command(
        #         'createIndexes',
        #         collection.name,
        #         indexes=[{
        #             'name': index_name,
        #             'key': {field_name: 'vector'},
        #             # Specific options for vector index type (e.g., 'ivfFlat', 'hnsw') and similarity
        #             # This part is highly dependent on the MongoDB version and vector search implementation
        #             'vectorSearch': {
        #                 'dimensions': vector_dimension,
        #                 'similarity': 'cosine', # or 'euclidean', 'dotProduct'
        #                 # 'type': 'ivfFlat', # Example type
        #                 # 'm': 16, # Example parameter for HNSW
        #                 # 'efConstruction': 100 # Example parameter for HNSW
        #             }
        #         }]
        #     )
        #     print(f"Vector index '{index_name}' created successfully on '{field_name}'.")
        # except Exception as e:
        #     if "already exists" in str(e).lower():
        #         print(f"Vector index '{index_name}' already exists on '{field_name}'.")
        #     else:
        #         raise e

    except Exception as e:
        print(
            f"Error during vector index creation check/attempt for field '{field_name}': {e}"
        )
        # It's often better to ensure indexes are created out-of-band (e.g., via deployment scripts or Atlas UI)
        # rather than at runtime in the application, especially for complex indexes like search/vector indexes.


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


def vector_search(
    collection_name,
    query_text,
    index_field="embedding",
    num_results=5,
    text_field_to_return=None,
):
    """Performs a vector search in the specified collection."""
    db = get_database()
    collection = db[collection_name]

    query_vector = generate_text_embedding(query_text)

    if not query_vector or all(
        v == 0.0 for v in query_vector
    ):  # Check if embedding failed or is zero vector
        print("Could not generate a valid query vector.")
        return pd.DataFrame()  # Return empty DataFrame

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",  # This should match the name of your Atlas Search vector index
                "path": index_field,
                "queryVector": query_vector,
                "numCandidates": num_results * 10,  # Number of candidates to consider
                "limit": num_results,
            }
        },
        {
            "$project": {
                "_id": 0,
                "score": {"$meta": "vectorSearchScore"},  # Include the search score
                # Add other fields you want to return
            }
        },
    ]

    # If a specific text field should be returned, add it to $project
    if text_field_to_return and text_field_to_return != "_id":
        pipeline[1]["$project"][text_field_to_return] = 1
    else:  # If no specific text field, try to return all fields except embedding
        # This requires knowing the fields. A safer bet is to project specific known fields.
        # For simplicity, if not specified, we are only returning score.
        # You might want to list all fields of the document excluding the 'embedding' field.
        # Example: Get all fields except 'embedding'
        # This is complex to do dynamically in $project without knowing field names.
        # A common pattern is to return specific, known, useful fields.
        pass  # Default projection only includes score and _id (if not excluded)

    try:
        results = list(collection.aggregate(pipeline))
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error during vector search: {e}")
        print(
            "Please ensure that a vector search index named 'vector_index' exists on the collection "
            f"'{collection_name}' for the field '{index_field}' and that the query is valid."
        )
        return pd.DataFrame()  # Return empty DataFrame on error
