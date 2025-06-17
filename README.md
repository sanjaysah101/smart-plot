# AI-Powered Data Visualisation Tool

This project aims to build an AI-Powered Data Visualisation Tool for public datasets. It creates an interactive dashboard that utilizes Google Cloud AI to extract insights and generate charts from a public dataset, with MongoDB serving as the backend for storing processed data.

## Project Setup and Running Instructions

To set up and run this project, please follow these steps:

### Prerequisites

- Python 3.8+
- `uv` (a fast Python package installer and resolver)
- MongoDB instance (local or cloud-hosted)
- Google Cloud Platform account with necessary AI services enabled

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-repo/data-viz-ai.git
    cd data-viz-ai
    ```

2. **Install dependencies using `uv`:**

    ```bash
    uv sync
    ```

    or

    ```bash
    uv add -r .\requirements.txt 
    ````

### Configuration

Create a `.env` file in the root directory of the project and add your configuration variables. For example:

```bash
MONGODB_URI="mongodb://localhost:27017/"
GOOGLE_CLOUD_PROJECT_ID="your-gcp-project-id"
GOOGLE_CLOUD_API_KEY="your-gcp-api-key"
```

Replace `mongodb://localhost:27017/` with your MongoDB connection string and `your-gcp-project-id` and `your-gcp-api-key` with your Google Cloud Platform project ID and API key, respectively.

### Running the Application

1. **Start the MongoDB instance:**

    ```bash
    mongod
    ```

2. **Run the Streamlit application:**

    ```bash
    streamlit run main.py
    or,
    uv run -- streamlit run main.py
    ```

3. **Access the application in your browser:**

    The application will be running at `http://localhost:8501`. You can access it in your web browser to start using the AI-powered data visualisation tool.

### Note

- This project assumes that you have the necessary permissions and access to the Google Cloud Platform project and the MongoDB instance.
- The Google Cloud API key should have the necessary permissions to access the Google Cloud AI services.
