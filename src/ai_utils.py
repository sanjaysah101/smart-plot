import os

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)


def get_data_insights(dataframe, specific_columns=None, question=None):
    """Generate insights from a dataframe using Gemini AI"""
    # Create a model instance
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

    # Prepare the dataframe information
    df_info = dataframe.describe().to_string()
    df_head = dataframe.head(5).to_string()

    # Prepare column information if specific columns are provided
    column_info = ""
    if specific_columns:
        for col in specific_columns:
            if col in dataframe.columns:
                column_info += f"\nColumn {col} statistics:\n"
                column_info += dataframe[col].describe().to_string()

    # Prepare the prompt
    if question:
        prompt = f"""I have a dataset with the following statistics:
        {df_info}
        
        Here's a sample of the data:
        {df_head}
        
        {column_info}
        
        Based on this data, please answer the following question: {question}
        Provide a detailed analysis with key insights."""
    else:
        prompt = f"""I have a dataset with the following statistics:
        {df_info}
        
        Here's a sample of the data:
        {df_head}
        
        {column_info}
        
        Based on this data, please provide:
        1. A summary of the key patterns and trends
        2. Interesting insights or anomalies
        3. Suggestions for visualizations that would best represent this data
        4. Potential research questions that could be explored
        
        Format your response in markdown with clear sections."""

    # Generate the response
    response = model.generate_content(prompt)
    return response.text


def generate_text_embedding(text_to_embed):
    """Generates an embedding for the given text using Gemini AI."""
    if not text_to_embed or not isinstance(text_to_embed, str):
        # Return None or a zero vector if the input is not suitable.
        # The size of the zero vector should match the embedding dimension.
        # For 'models/text-embedding-004', the dimension is 768.
        # Alternatively, raise an error or handle as appropriate.
        return [0.0] * 768 # Or None, depending on how you want to handle it downstream

    try:
        # Using a specific model for embeddings, e.g., 'models/text-embedding-004'
        # Ensure this model is available and appropriate for your use case.
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text_to_embed,
            task_type="RETRIEVAL_DOCUMENT" # or RETRIEVAL_QUERY, SEMANTIC_SIMILARITY etc.
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return a zero vector or None in case of an error to avoid breaking the pipeline
        return [0.0] * 768 # Or None
