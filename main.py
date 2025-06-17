import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.ai_utils import get_data_insights

# Import our custom modules
from src.db_utils import get_dataset, get_dataset_names, store_dataset
from src.kaggle_utils import download_dataset, search_datasets

# Load environment variables
load_dotenv()

# Page configuration
PAGE_CONFIG = {
    "page_title": "AI-Powered Data Visualization",
    "page_icon": "chart_with_upwards_trend:",
    "layout": "centered",
}
st.set_page_config(**PAGE_CONFIG)

# Initialize session state variables
for key in ["df", "filename", "option", "opt", "columnList", "insights", "data_loaded"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Initialize data_loaded as False if not set
if st.session_state.data_loaded is None:
    st.session_state.data_loaded = False


def paginated_dataframe(df, page_size=100):
    total_rows = len(df)
    page_num = st.number_input("Page", 1, (total_rows // page_size) + 1)
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    st.dataframe(df.iloc[start_idx:end_idx])


def sidebar():
    """Sidebar navigation and data source handling"""
    with st.sidebar:
        st.title("Data Source")

        # Create tabs for different data sources
        data_source = st.radio(
            "Select Data Source",
            ["Upload File", "Kaggle Dataset", "MongoDB Storage"],
            key="data_source_radio",
        )

        # Handle different data sources
        if data_source == "Upload File":
            handle_file_upload()
        elif data_source == "Kaggle Dataset":
            handle_kaggle_dataset()
        elif data_source == "MongoDB Storage":
            handle_mongodb_storage()

        # Show column and chart options if data is loaded
        show_data_options()


def show_data_options():
    """Show column selection and chart options - always visible when data is loaded"""
    if st.session_state.data_loaded and st.session_state.columnList:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Data Options")

        # Column selection
        st.session_state.option = st.sidebar.selectbox(
            "Select Column",
            st.session_state.columnList,
            key="main_column_selectbox",
        )

        # Chart type selection
        st.sidebar.subheader("Chart Type")
        graph_options = ["Line Chart", "Bar Chart", "Pie Chart"]
        st.session_state.opt = st.sidebar.radio(
            "Select chart type", graph_options, key="main_graph_type_radio"
        )


def handle_file_upload():
    """Handle file upload data source"""
    allowedExtension = ["csv", "xlsx"]

    uploaded_file = st.sidebar.file_uploader(
        label="Upload your csv or excel file (200 MB Max).",
        type=allowedExtension,
        key="file_uploader",
    )

    if uploaded_file is not None:
        filename = uploaded_file.name
        extension = filename[filename.index(".") + 1 :]
        filename = filename[: filename.index(".")]

        if extension in allowedExtension:
            try:
                df = (
                    pd.read_csv(uploaded_file)
                    if extension == "csv"
                    else pd.read_excel(uploaded_file)
                )

                # Store in session state
                st.session_state.df = df
                st.session_state.filename = filename
                st.session_state.columnList = df.columns.values.tolist()
                st.session_state.data_loaded = True

                st.sidebar.success(f"File '{filename}' loaded successfully!")

                # Option to save to MongoDB
                if st.sidebar.button("Save to MongoDB", key="save_to_mongo"):
                    num_records = store_dataset(filename, df)
                    st.sidebar.success(
                        f"Successfully stored {num_records} records in MongoDB"
                    )
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
        else:
            st.sidebar.error("File Format is not supported")


def handle_kaggle_dataset():
    """Handle Kaggle dataset data source"""
    # Check if Kaggle API credentials exist
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")

    if not os.path.exists(kaggle_json_path):
        st.sidebar.warning(
            "Kaggle API credentials not found. Please create a kaggle.json file in ~/.kaggle/ "
            "with your Kaggle username and API key."
        )
        st.sidebar.markdown(
            "[How to get Kaggle API credentials](https://github.com/Kaggle/kaggle-api#api-credentials)"
        )
        return

    # Search for datasets
    search_query = st.sidebar.text_input(
        "Search for datasets", "covid", key="kaggle_search_input"
    )

    if st.sidebar.button("Search", key="kaggle_search_btn"):
        with st.spinner("Searching for datasets..."):
            try:
                datasets = search_datasets(search_query)
                st.session_state.kaggle_datasets = datasets
                st.session_state.kaggle_dataset_refs = [ds.ref for ds in datasets]
                st.session_state.kaggle_dataset_titles = [ds.title for ds in datasets]
            except Exception as e:
                st.sidebar.error(f"Error searching for datasets: {e}")

    # Select and download a dataset
    if (
        "kaggle_dataset_titles" in st.session_state
        and st.session_state.kaggle_dataset_titles
    ):
        selected_dataset_idx = st.sidebar.selectbox(
            "Select a dataset",
            range(len(st.session_state.kaggle_dataset_titles)),
            format_func=lambda x: st.session_state.kaggle_dataset_titles[x],
            key="kaggle_dataset_selectbox",
        )

        if st.sidebar.button("Download and Load", key="download_and_load_btn"):
            with st.spinner("Downloading dataset..."):
                try:
                    dataset_ref = st.session_state.kaggle_dataset_refs[
                        selected_dataset_idx
                    ]
                    df, filename = download_dataset(dataset_ref)

                    if df is not None:
                        # Store in session state
                        st.session_state.df = df
                        st.session_state.filename = filename.split(".")[0]
                        st.session_state.columnList = df.columns.values.tolist()
                        st.session_state.data_loaded = True

                        # Option to save to MongoDB
                        num_records = store_dataset(st.session_state.filename, df)
                        st.sidebar.success(
                            f"Dataset loaded and {num_records} records stored in MongoDB"
                        )
                    else:
                        st.sidebar.error("No CSV files found in the dataset")
                except Exception as e:
                    st.sidebar.error(f"Error downloading dataset: {e}")


def handle_mongodb_storage():
    """Handle MongoDB storage data source"""
    # Get list of datasets from MongoDB
    try:
        dataset_names = get_dataset_names()

        if not dataset_names:
            st.sidebar.info(
                "No datasets found in MongoDB. Please upload or download a dataset first."
            )
            return

        selected_dataset = st.sidebar.selectbox(
            "Select a dataset", dataset_names, key="mongo_dataset_selectbox"
        )

        if st.sidebar.button("Load Dataset", key="load_dataset_btn"):
            with st.spinner("Loading dataset from MongoDB..."):
                try:
                    df = get_dataset(selected_dataset)

                    # Store in session state
                    st.session_state.df = df
                    st.session_state.filename = selected_dataset
                    st.session_state.columnList = df.columns.values.tolist()
                    st.session_state.data_loaded = True

                    st.sidebar.success(
                        f"Dataset '{selected_dataset}' loaded successfully!"
                    )
                except Exception as e:
                    st.sidebar.error(f"Error loading dataset: {e}")
    except Exception as e:
        st.sidebar.error(f"Error connecting to MongoDB: {e}")


def getIndexes(columnName, value):
    """Get the index of a value in a column"""
    count = -1
    for i in st.session_state.df[columnName]:
        count += 1
        if i == value:
            return count
    return -1


def generate_ai_insights():
    """Generate AI insights using Gemini"""
    if st.session_state.df is not None:
        st.subheader("AI Insights")

        # Allow user to ask specific questions
        question = st.text_input(
            "Ask a question about the data (optional)", key="ai_question_input"
        )

        # Allow user to select specific columns for analysis
        selected_columns = st.multiselect(
            "Select columns for detailed analysis (optional)",
            st.session_state.columnList,
            key="ai_column_multiselect",
        )

        if st.button("Generate Insights", key="generate_insights_btn"):
            with st.spinner("Generating insights with Gemini AI..."):
                try:
                    insights = get_data_insights(
                        st.session_state.df,
                        specific_columns=selected_columns if selected_columns else None,
                        question=question if question else None,
                    )
                    st.session_state.insights = insights
                except Exception as e:
                    st.error(f"Error generating insights: {e}")

        # Display insights if available
        if st.session_state.insights:
            st.markdown(st.session_state.insights)


def mainContent():
    st.title("AI-Powered Data Visualization")
    st.markdown(
        """This tool allows you to visualize data from various sources, store it in MongoDB, 
        and generate AI-powered insights using Google's Gemini AI."""
    )

    if not st.session_state.data_loaded:
        st.info("Please select and load a data source from the sidebar to begin.")
        return

    # Display tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Data Explorer", "Visualization", "AI Insights"])

    with tab1:
        st.header("Data Explorer")
        paginated_dataframe(st.session_state.df)

        # Display basic statistics
        if st.checkbox("Show Statistics", key="show_stats_checkbox"):
            st.write(st.session_state.df.describe())

    with tab2:
        st.header("Visualize Your Data")

        # Column selection for comparison
        st.subheader("Choose the Column To which you want to compare")

        # Data selection
        selectOption = []
        if (
            st.session_state.columnList is not None
            and len(st.session_state.columnList) > 0
        ):
            for i in st.session_state.df[st.session_state.columnList[0]]:
                selectOption.append(i)

            selectedData = st.multiselect(
                f"Choose {st.session_state.columnList[0]} to see",
                selectOption,
                key="selected_data_multiselect",
            )

            # Prepare data for visualization
            dataToVisualize = []
            for i in selectedData:
                index = getIndexes(st.session_state.columnList[0], i)
                if index >= 0:
                    value = st.session_state.df[st.session_state.option][index]
                    if type(value) is not str:
                        dataToVisualize.append(value)
                    else:
                        st.warning(f"The data type of {value} is not supported")

            # Generate the selected chart
            if st.session_state.opt == "Line Chart":
                st.subheader(f"Line Chart for {st.session_state.filename}")
                if len(dataToVisualize) > 0:
                    st.line_chart(dataToVisualize)
                else:
                    st.info("No numeric data selected for visualization")

            elif st.session_state.opt == "Bar Chart":
                st.subheader(f"Bar Chart for {st.session_state.filename}")
                if len(dataToVisualize) > 0:
                    st.bar_chart(dataToVisualize)
                else:
                    st.info("No numeric data selected for visualization")

            elif st.session_state.opt == "Pie Chart":
                st.subheader(f"Pie Chart for {st.session_state.filename}")
                if len(dataToVisualize) > 0:
                    x = np.array(dataToVisualize, "f")
                    fig = plt.figure(figsize=(10, 10))
                    plt.pie(x, labels=selectedData, autopct="%.2f%%")
                    plt.legend(title=st.session_state.option)
                    st.pyplot(fig)
                else:
                    st.info("No numeric data selected for visualization")

    with tab3:
        generate_ai_insights()


if __name__ == "__main__":
    sidebar()
    mainContent()
