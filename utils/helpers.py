import streamlit as st
import pandas as pd
import plotly.io as pio

# Available models configuration
MODEL_OPTIONS = {
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3 8B": "llama3-8b-8192",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 7B": "gemma-7b-it"
}

# Visualization types
VIZ_TYPES = ["histogram", "scatter", "bar", "line", "box", "heatmap"]

# Example queries for display
EXAMPLE_QUERIES = [
    "What is the average value of [column]?",
    "Show me the distribution of [column].",
    "Is there a correlation between [column1] and [column2]?",
    "What are the top 5 values in [column]?",
    "Create a visualization of [column].",
    "Identify any outliers in the data.",
    "What insights can you provide about this dataset?",
    "Create a scatter plot comparing [column1] and [column2].",
    "How does [column] vary over time?",
    "What are the key factors influencing [column]?"
]

def setup_page_config():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="AI Data Analyst",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"  # Hide sidebar by default
    )

def load_custom_css():
    """Load custom CSS styles for a minimal UI."""
    st.markdown("""
        <style>
        /* Global Styles */
        .main {
            padding: 1.5rem;
            max-width: 1200px;
            margin: 0 auto;
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide sidebar completely */
        .css-1d391kg, .css-1wrcr25 {
            display: none !important;
        }
        
        /* Full width for main content */
        .block-container {
            max-width: 1200px;
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Headers */
        h1 {
            color: #1E3A8A;
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px !important;
            line-height: 1.2 !important;
            margin-bottom: 1rem !important;
        }
        
        h2 {
            color: #2563EB;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        h3 {
            color: #3B82F6;
            font-size: 1.3rem !important;
            font-weight: 600 !important;
            margin-top: 1.25rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            transition: all 0.2s ease;
            border: none;
            background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
            color: white;
        }
        
        /* File Uploader */
        .stFileUploader {
            border: 2px dashed #60A5FA;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
        }
        
        /* DataFrames */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid #E5E7EB;
        }
        
        .dataframe thead th {
            background-color: #F3F4F6;
            padding: 0.75rem !important;
            font-weight: 600;
            color: #1F2937;
        }
        
        /* Plotly Charts - more detailed styling */
        .js-plotly-plot {
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            padding: 0.5rem;
            background: white;
            margin: 1rem 0;
        }
        
        .js-plotly-plot .main-svg {
            border-radius: 6px;
        }
        
        /* Compact layout for inputs */
        .stTextInput, .stSelectbox, .stTextArea {
            margin-bottom: 1rem;
        }
        
        </style>
    """, unsafe_allow_html=True)

def display_data_preview(df):
    """Display a minimal preview of the dataframe."""
    # Get basic dataframe info
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Basic dataset info in a compact row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", format_number(total_rows))
    with col2:
        st.metric("Columns", total_cols)
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    
    # Data sample in an expander
    with st.expander("Data Sample", expanded=True):
        # Calculate cell background colors based on values
        def highlight_cols(s):
            if pd.api.types.is_numeric_dtype(s):
                # Normalize to [0, 1] range for numeric columns
                min_val = s.min()
                max_val = s.max()
                if min_val != max_val:  # Avoid division by zero
                    normalized = (s - min_val) / (max_val - min_val)
                    return ['background-color: rgba(37, 99, 235, {})'.format(0.1 + 0.3*v) for v in normalized]
            # For non-numeric or single-value columns
            return ['background-color: rgba(37, 99, 235, 0.05)'] * len(s)
        
        # Style the dataframe
        styled_df = df.head(5).style.apply(highlight_cols)
        
        # Display styled dataframe
        st.dataframe(styled_df, use_container_width=True)
    
    # Column information
    with st.expander("Column Information", expanded=False):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count(),
            'Null': df.isnull().sum(),
            'Unique': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

def format_number(num):
    """Format a number with K, M, B suffixes."""
    if num < 1000:
        return str(num)
    
    for unit in ['K', 'M', 'B', 'T']:
        num /= 1000
        if num < 1000:
            return f"{num:.1f}{unit}"
    
    return f"{num:.1f}T" 