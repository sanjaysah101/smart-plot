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
