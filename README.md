# Data-viz-ai - AI-Powered Data Visualization Tool

Data-viz-ai is an interactive, AI-enhanced data visualization tool built using Streamlit, integrated with Google Cloud AI for automated insight extraction and MongoDB for data storage.

## 🚀 Features

- 📂 Upload CSV or Excel files, or import datasets from Kaggle or MongoDB
- 🔍 Use Google Cloud AI to analyze datasets
- 📊 Generate insightful charts dynamically
- 💾 Persist data and visualizations using MongoDB
- 🧠 Sidebar state persistence for smooth user experience

## 🧰 Project Setup and Running Instructions

### ✅ Prerequisites

- Python 3.12+
- `uv` (a fast Python package installer and resolver)
- MongoDB instance (local or cloud-hosted)
- Google Cloud Platform account with AI services enabled

### 📦 Installation

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
    uv add -r requirements.txt
    ```

### ⚙️ Configuration

Create a `.env` file in the project root with the following:

```env
MONGODB_URI="mongodb://localhost:27017/"
GOOGLE_CLOUD_PROJECT_ID="your-gcp-project-id"
GOOGLE_CLOUD_API_KEY="your-gcp-api-key"
```

Update values with your MongoDB URI, GCP project ID, and API key.

### ▶️ Running the Application

1. **Ensure MongoDB is running:**

    ```bash
    mongod
    ```

2. **Start the Streamlit app:**

    ```bash
    streamlit run main.py
    # or
    uv run -- streamlit run main.py
    ```

3. **Open in browser:**

    Visit [http://localhost:8501](http://localhost:8501) to start using the tool.

## 📌 Notes

- Make sure your GCP API key has permissions to access the required AI services.
- Sidebar remembers your last state and selection for a seamless workflow.
- Ensure Kaggle credentials (`~/.kaggle/kaggle.json`) are set up for dataset import.

## 📄 License

MIT License. See `LICENSE` file for more details.

## 🙋‍♀️ Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Happy Visualizing! 🚀
