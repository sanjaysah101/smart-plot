import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.ai_utils import get_data_insights

# Import our custom modules
from src.db_utils import (
    get_dataset,
    get_dataset_names,
    store_dataset,
    vector_search,  # Added import
)
from src.kaggle_utils import download_dataset, search_datasets

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
            handle_uploaded_file()  # Changed from handle_file_upload()
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

        # Chart type selection
        st.sidebar.subheader("Chart Type")
        graph_options = ["Line Chart", "Bar Chart", "Pie Chart"]
        st.session_state.opt = st.sidebar.radio(
            "Select chart type", graph_options, key="main_graph_type_radio"
        )


# def handle_file_upload():
#     """Handle file upload data source"""
#     allowedExtension = ["csv", "xlsx"]

#     uploaded_file = st.sidebar.file_uploader(
#         label="Upload your csv or excel file (200 MB Max).",
#         type=allowedExtension,
#         key="file_uploader",
#     )

#     if uploaded_file is not None:
#         filename = uploaded_file.name
#         extension = filename[filename.index(".") + 1 :]
#         filename = filename[: filename.index(".")]

#         if extension in allowedExtension:
#             try:
#                 df = (
#                     pd.read_csv(uploaded_file)
#                     if extension == "csv"
#                     else pd.read_excel(uploaded_file)
#                 )

#                 # Store in session state
#                 st.session_state.df = df
#                 st.session_state.filename = filename
#                 st.session_state.columnList = df.columns.values.tolist()
#                 st.session_state.data_loaded = True

#                 st.sidebar.success(f"File '{filename}' loaded successfully!")

#                 # Option to save to MongoDB
#                 if st.sidebar.button("Save to MongoDB", key="save_to_mongo"):
#                     num_records = store_dataset(filename, df)
#                     st.sidebar.success(
#                         f"Successfully stored {num_records} records in MongoDB"
#                     )
#             except Exception as e:
#                 st.sidebar.error(f"Error loading file: {e}")
#         else:
#             st.sidebar.error("File Format is not supported")


def handle_uploaded_file():
    """Handle uploaded file data source"""
    allowedExtension = ["csv", "xlsx"]  # Added from old handle_file_upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV or Excel file",
        type=allowedExtension,
        key="file_uploader",  # Use allowedExtension
    )
    if uploaded_file is not None:
        # Logic from old handle_file_upload to read file and set initial session state
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
                st.session_state.df = df
                st.session_state.filename = filename
                st.session_state.columnList = df.columns.values.tolist()
                st.session_state.data_loaded = True
                st.sidebar.success(f"File '{filename}' loaded successfully!")

                # Ask user which column to use for text embedding, or to combine columns
                st.sidebar.subheader("Configure Text for Embedding")
                embedding_text_column_options = (
                    ["None (Skip Embedding)"]
                    + st.session_state.columnList
                    + ["Combine multiple columns"]
                )

                selected_embedding_source = st.sidebar.selectbox(
                    "Select column for text embedding (or combine)",
                    embedding_text_column_options,
                    key="embedding_source_select",
                )

                combined_text_column_name = "combined_text_for_embedding"
                text_column_for_embedding = None  # Initialize

                if selected_embedding_source == "Combine multiple columns":
                    columns_to_combine = st.sidebar.multiselect(
                        "Select columns to combine for embedding",
                        st.session_state.columnList,
                        key="columns_to_combine_multiselect",
                    )
                    if columns_to_combine:
                        df[combined_text_column_name] = (
                            df[columns_to_combine].astype(str).agg(" ".join, axis=1)
                        )
                        text_column_for_embedding = combined_text_column_name
                        st.session_state.last_embedded_text_column = (
                            combined_text_column_name  # Store for later default
                        )
                        st.sidebar.info(
                            f"Using combined column '{combined_text_column_name}' for embeddings."
                        )
                    else:
                        st.sidebar.warning(
                            "Please select columns to combine or choose a single column."
                        )
                        text_column_for_embedding = None
                elif selected_embedding_source != "None (Skip Embedding)":
                    text_column_for_embedding = selected_embedding_source
                    st.session_state.last_embedded_text_column = (
                        text_column_for_embedding  # Store for later default
                    )
                    st.sidebar.info(
                        f"Using column '{text_column_for_embedding}' for embeddings."
                    )
                else:
                    text_column_for_embedding = None
                    st.sidebar.info("Skipping text embedding.")

                # Option to save to MongoDB
                if st.sidebar.button("Save to MongoDB", key="save_to_mongodb_btn"):
                    num_records = store_dataset(
                        filename,
                        df,
                        text_column_for_embedding=text_column_for_embedding,
                    )
                    embedding_msg = (
                        "Embeddings generated."
                        if text_column_for_embedding
                        else "No embeddings generated."
                    )
                    st.sidebar.success(
                        f"Dataset saved to MongoDB with {num_records} records. {embedding_msg}"
                    )
                    if text_column_for_embedding:
                        st.sidebar.markdown(
                            "**Important:** For vector search to work, ensure you have a **vector search index** named `vector_index` on the `embedding` field in your MongoDB Atlas collection. Refer to the `create_vector_index` logs or Atlas documentation for setup details."
                        )

                    # If combined column was created and we don't want to persist it in the displayed df
                    if (
                        text_column_for_embedding == combined_text_column_name
                        and combined_text_column_name in df.columns
                    ):
                        df.drop(columns=[combined_text_column_name], inplace=True)
                        st.session_state.columnList = (
                            df.columns.values.tolist()
                        )  # Update column list if temp col dropped
            except Exception as e:
                st.sidebar.error(f"Error processing file: {e}")
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
                        st.session_state.df = df
                        st.session_state.filename = filename.split(".")[0]
                        st.session_state.columnList = df.columns.values.tolist()
                        st.session_state.data_loaded = True

                        # Ask user which column to use for text embedding
                        st.sidebar.subheader("Configure Text for Embedding (Kaggle)")
                        embedding_text_column_options_kaggle = (
                            ["None (Skip Embedding)"]
                            + st.session_state.columnList
                            + ["Combine multiple columns"]
                        )
                        selected_embedding_source_kaggle = st.sidebar.selectbox(
                            "Select column for text embedding (or combine)",
                            embedding_text_column_options_kaggle,
                            key="embedding_source_select_kaggle",
                        )

                        combined_text_column_name_kaggle = (
                            "combined_text_for_embedding_kaggle"
                        )
                        text_column_for_embedding_kaggle = None

                        if (
                            selected_embedding_source_kaggle
                            == "Combine multiple columns"
                        ):
                            columns_to_combine_kaggle = st.sidebar.multiselect(
                                "Select columns to combine for embedding",
                                st.session_state.columnList,
                                key="columns_to_combine_multiselect_kaggle",
                            )
                            if columns_to_combine_kaggle:
                                df[combined_text_column_name_kaggle] = (
                                    df[columns_to_combine_kaggle]
                                    .astype(str)
                                    .agg(" ".join, axis=1)
                                )
                                text_column_for_embedding_kaggle = (
                                    combined_text_column_name_kaggle
                                )
                                st.sidebar.info(
                                    f"Using combined column '{combined_text_column_name_kaggle}' for embeddings."
                                )
                            else:
                                st.sidebar.warning(
                                    "Please select columns to combine or choose a single column."
                                )
                        elif (
                            selected_embedding_source_kaggle != "None (Skip Embedding)"
                        ):
                            text_column_for_embedding_kaggle = (
                                selected_embedding_source_kaggle
                            )
                            st.sidebar.info(
                                f"Using column '{text_column_for_embedding_kaggle}' for embeddings."
                            )
                        else:
                            st.sidebar.info(
                                "Skipping text embedding for Kaggle dataset."
                            )

                        # Option to save to MongoDB
                        num_records = store_dataset(
                            st.session_state.filename,
                            df,
                            text_column_for_embedding=text_column_for_embedding_kaggle,
                        )
                        st.sidebar.success(
                            f"Dataset loaded and {num_records} records stored in MongoDB. "
                            f"{'Embeddings generated.' if text_column_for_embedding_kaggle else 'No embeddings generated.'}"
                        )
                        # If combined column was created and we don't want to persist it in the displayed df
                        if (
                            text_column_for_embedding_kaggle
                            == combined_text_column_name_kaggle
                            and combined_text_column_name_kaggle in df.columns
                        ):
                            df.drop(
                                columns=[combined_text_column_name_kaggle], inplace=True
                            )
                            st.session_state.columnList = df.columns.values.tolist()
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


def get_sample_data_for_viz(df, max_unique_values=100, sample_size=10000):
    """Get a sample of data for visualization to handle large datasets efficiently"""
    # If dataset is large, sample it first
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        st.info(
            f"Dataset is large ({len(df):,} rows). Using a sample of {sample_size:,} rows for visualization."
        )
    else:
        df_sample = df

    return df_sample


def prepare_visualization_data(df, x_column, y_column, selected_values):
    """Efficiently prepare data for visualization"""
    try:
        # Filter data based on selected values
        filtered_df = df[df[x_column].isin(selected_values)]

        # Group by x_column and aggregate y_column
        if pd.api.types.is_numeric_dtype(filtered_df[y_column]):
            # For numeric data, calculate mean
            viz_data = filtered_df.groupby(x_column)[y_column].mean().reset_index()
            return viz_data[x_column].tolist(), viz_data[y_column].tolist()
        else:
            # For non-numeric data, count occurrences
            viz_data = filtered_df.groupby(x_column)[y_column].count().reset_index()
            return viz_data[x_column].tolist(), viz_data[y_column].tolist()
    except Exception as e:
        st.error(f"Error preparing visualization data: {e}")
        return [], []


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
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Data Explorer", "Visualization", "AI Insights", "Vector Search"]
    )  # Added Vector Search Tab

    with tab1:
        st.header("Data Explorer")

        # Show dataset info
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(st.session_state.df):,}")
        with col2:
            st.metric("Columns", len(st.session_state.df.columns))
        with col3:
            st.metric(
                "Memory Usage",
                f"{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
            )

        paginated_dataframe(st.session_state.df)

        # Display basic statistics
        if st.checkbox("Show Statistics", key="show_stats_checkbox"):
            st.write(st.session_state.df.describe())

    with tab2:
        st.header("Visualize Your Data")

        if len(st.session_state.df) == 0:
            st.warning("No data available for visualization.")
            return

        # Column selection for comparison
        st.subheader("Configure Visualization")

        # Get sample data for large datasets
        viz_df = get_sample_data_for_viz(st.session_state.df)

        # Select X and Y columns
        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox(
                "Select X-axis column (categories)",
                st.session_state.columnList,
                key="x_column_select",
            )
        with col2:
            y_column = st.selectbox(
                "Select Y-axis column (values)",
                st.session_state.columnList,
                index=st.session_state.columnList.index(st.session_state.option)
                if st.session_state.option in st.session_state.columnList
                else 0,
                key="y_column_select",
            )

        # Get unique values for X column (with limit for performance)
        try:
            unique_values = viz_df[x_column].value_counts().head(50).index.tolist()

            if len(viz_df[x_column].unique()) > 50:
                st.info(
                    f"Showing top 50 most frequent values from {len(viz_df[x_column].unique())} unique values in '{x_column}'"
                )

            selectedData = st.multiselect(
                f"Choose values from '{x_column}' to visualize",
                unique_values,
                default=unique_values[:10]
                if len(unique_values) >= 10
                else unique_values,
                key="selected_data_multiselect",
            )

            if not selectedData:
                st.warning("Please select at least one value to visualize.")
                return

            # Prepare data for visualization
            with st.spinner("Preparing visualization data..."):
                labels, values = prepare_visualization_data(
                    viz_df, x_column, y_column, selectedData
                )

            if not values:
                st.warning("No valid data found for the selected values.")
                return

            # Generate the selected chart
            if st.session_state.opt == "Line Chart":
                st.subheader(f"Line Chart: {y_column} by {x_column}")
                if len(values) > 0:
                    chart_data = pd.DataFrame({x_column: labels, y_column: values})
                    st.line_chart(chart_data.set_index(x_column))
                else:
                    st.info("No numeric data available for line chart")

            elif st.session_state.opt == "Bar Chart":
                st.subheader(f"Bar Chart: {y_column} by {x_column}")
                if len(values) > 0:
                    chart_data = pd.DataFrame({x_column: labels, y_column: values})
                    st.bar_chart(chart_data.set_index(x_column))
                else:
                    st.info("No numeric data available for bar chart")

            elif st.session_state.opt == "Pie Chart":
                st.subheader(f"Pie Chart: {y_column} by {x_column}")
                if len(values) > 0 and all(
                    isinstance(val, (int, float)) and val >= 0 for val in values
                ):
                    fig, ax = plt.subplots(figsize=(10, 8))
                    wedges, texts, autotexts = ax.pie(
                        values, labels=labels, autopct="%1.1f%%", startangle=90
                    )
                    ax.set_title(f"{y_column} by {x_column}")

                    # Improve readability for many labels
                    if len(labels) > 8:
                        ax.legend(
                            wedges,
                            labels,
                            title=x_column,
                            loc="center left",
                            bbox_to_anchor=(1, 0, 0.5, 1),
                        )
                        plt.setp(texts, visible=False)

                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Pie chart requires positive numeric values")

        except Exception as e:
            st.error(f"Error creating visualization: {e}")
            st.info(
                "Try selecting different columns or reducing the number of selected values."
            )

    with tab3:
        generate_ai_insights()

    with tab4:  # New Tab for Vector Search
        st.header("Semantic Search with AI Embeddings")

        if (
            not st.session_state.data_loaded
            or st.session_state.df is None
            or st.session_state.filename is None
        ):
            st.info(
                "Please load a dataset and ensure it has been stored in MongoDB with embeddings to use vector search."
            )
            return

        # Input for search query
        search_query = st.text_input(
            "Enter your search query:", key="vector_search_query"
        )

        # Select which field's text content to return alongside the search results
        # This assumes that the original text field used for embedding (or another relevant text field) is known
        # and present in the stored documents.
        # For simplicity, let's allow choosing from existing columns, though the actual embedded text might be a combination.
        # This part might need refinement based on how text_column_for_embedding was handled during store_dataset.

        # Try to guess the original text column if it was a single column, or default
        # This is a heuristic. A more robust way would be to store metadata about the embedded field.
        default_text_field = st.session_state.get("last_embedded_text_column", None)
        if default_text_field and default_text_field not in st.session_state.columnList:
            default_text_field = (
                None  # Reset if not in current columns (e.g. combined temp column)
            )
        if not default_text_field and st.session_state.columnList:
            # Heuristic: pick the first string-like column if available, or just the first column
            first_str_col = next(
                (
                    col
                    for col in st.session_state.columnList
                    if st.session_state.df[col].dtype == "object"
                ),
                None,
            )
            default_text_field = (
                first_str_col if first_str_col else st.session_state.columnList[0]
            )

        available_fields_for_return = [
            col for col in st.session_state.columnList if col != "embedding"
        ]  # Exclude embedding itself

        text_field_to_return = None
        if available_fields_for_return:
            text_field_to_return = st.selectbox(
                "Select a text field to display from search results (optional):",
                options=["None (Show only score)"] + available_fields_for_return,
                index=(available_fields_for_return.index(default_text_field) + 1)
                if default_text_field
                and default_text_field in available_fields_for_return
                else 0,
                key="vector_search_return_field",
            )
            if text_field_to_return == "None (Show only score)":
                text_field_to_return = None
        else:
            st.info(
                "No suitable text fields available in the current dataset to display with search results."
            )

        num_results = st.slider(
            "Number of results to retrieve:",
            min_value=1,
            max_value=20,
            value=5,
            key="vector_search_num_results",
        )

        if st.button("Search", key="vector_search_button"):
            if not search_query:
                st.warning("Please enter a search query.")
                return

            with st.spinner("Performing vector search..."):
                try:
                    # Assuming 'embedding' is the field where vectors are stored
                    # And 'vector_index' is the name of the search index in Atlas
                    search_results_df = vector_search(
                        st.session_state.filename,
                        search_query,
                        index_field="embedding",
                        num_results=num_results,
                        text_field_to_return=text_field_to_return,
                    )

                    if not search_results_df.empty:
                        st.subheader("Search Results")
                        st.dataframe(search_results_df)
                    else:
                        st.info(
                            "No results found, or an error occurred during the search. "
                            "Ensure the dataset was stored with embeddings and the vector index is configured in MongoDB Atlas."
                        )
                except Exception as e:
                    st.error(f"Error during vector search: {e}")
                    st.error(
                        "Please ensure your MongoDB Atlas instance is configured for vector search and the index 'vector_index' exists."
                    )


if __name__ == "__main__":
    sidebar()
    mainContent()
