import os
import logging
import sys
from io import StringIO
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import re
import traceback

from models.analyst import DataAnalyst
from utils.helpers import (
    MODEL_OPTIONS,
    VIZ_TYPES,
    EXAMPLE_QUERIES,
    setup_page_config,
    load_custom_css,
    display_data_preview,
    format_number
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("DataAnalysisApp")

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_theme' not in st.session_state:
        st.session_state.current_theme = "Blue"
    if 'analyst' not in st.session_state:
        st.session_state.analyst = None

def display_welcome():
    """Display a minimal welcome message."""
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0 2rem 0;">
        <p style="font-size: 1.2rem; color: #4B5563; max-width: 800px; margin: 0 auto; line-height: 1.7;">
            Upload a dataset and ask questions in natural language to gain insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example queries in a more compact layout
    st.subheader("Example Queries")
    queries_html = ""
    for query in EXAMPLE_QUERIES:
        queries_html += f"""
        <div style="background: white; border-radius: 12px; padding: 0.75rem 1rem;
                    margin-bottom: 0.5rem; border: 1px solid #E5E7EB;">
            <p style="margin: 0; color: #4B5563;">{query}</p>
        </div>
        """
    
    st.markdown(queries_html, unsafe_allow_html=True)

def handle_file_upload():
    """Handle CSV file upload and processing."""
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your data file in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.analyst = DataAnalyst(
                df,
                model=MODEL_OPTIONS[st.session_state.get('selected_model', "Llama 3.3 70B")]
            )
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            logger.error(f"File upload error: {str(e)}")
            return None
    return None

def display_api_key_input():
    """Display API key input field."""
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Enter your Groq API key",
        key="api_key_input"
    )
    
    if api_key:
        st.session_state.api_key = api_key
    
    return api_key

def display_model_selection():
    """Display model selection dropdown."""
    selected_model = st.selectbox(
        "Select Model",
        list(MODEL_OPTIONS.values()),
        index=0,
        key="model_selection"
    )
    
    return selected_model

def display_analysis_interface(df):
    """Display a simplified analysis interface."""
    # Use a single-panel layout instead of tabs
    st.markdown("""
    <div style="margin: 1.5rem 0 1rem 0;">
        <h2 style="color: #1E3A8A; font-size: 1.5rem; margin-bottom: 0.5rem;">Ask a Question</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Text input for question
    query = st.text_area(
        "Enter your question about the data",
        height=80,
        placeholder="e.g., What is the correlation between column A and column B?",
        key="query_input"
    )
    
    # Process question
    if st.button("Analyze", key="analyze_button"):
        if query:
            with st.spinner("Analyzing data..."):
                try:
                    # Get the response
                    results = st.session_state.analyst.analyze(query)
                    
                    # Display results
                    display_analysis_results(results)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}, {traceback.format_exc()}")
        else:
            st.warning("Please enter a question.")
    
    # Quick visualization section
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h2 style="color: #1E3A8A; font-size: 1.5rem; margin-bottom: 0.5rem;">Quick Visualization</h2>
    </div>
    """, unsafe_allow_html=True)
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        viz_type = st.selectbox("Visualization Type", VIZ_TYPES, key="viz_type")
    
    with viz_col2:
        selected_columns = select_columns_for_viz(df, viz_type)
    
    if st.button("Generate Visualization", key="viz_button"):
        with st.spinner("Generating visualization..."):
            try:
                if not selected_columns:
                    st.warning("Please select at least one column.")
                else:
                    title = f"{viz_type.capitalize()} of {', '.join(selected_columns)}"
                    viz_result = st.session_state.analyst.generate_visualization(
                        viz_type=viz_type,
                        columns=selected_columns,
                        title=title
                    )
                    
                    if viz_result.get("visualizations"):
                        viz = viz_result["visualizations"][0]
                        if viz["type"] == "plotly":
                            with open(viz["path"], 'r') as f:
                                st.components.v1.html(f.read(), height=600)
                        else:
                            st.image(viz["path"], use_column_width=True)
                    else:
                        st.warning("No visualization generated.")
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
                logger.error(f"Visualization error: {str(e)}")

def select_columns_for_viz(df, viz_type):
    """Helper function to select columns based on visualization type."""
    if viz_type in ["scatter", "line"]:
        x_col = st.selectbox("X-axis Column", df.columns, key="x_column")
        y_col = st.selectbox("Y-axis Column", df.columns, key="y_column")
        return [x_col, y_col]
    elif viz_type == "heatmap":
        return st.multiselect("Select Columns", df.columns, key="heatmap_columns")
    else:
        col = st.selectbox("Select Column", df.columns, key="primary_column")
        if viz_type == "bar":
            include_secondary = st.checkbox("Include Secondary Column")
            if include_secondary:
                secondary_col = st.selectbox(
                    "Secondary Column",
                    [c for c in df.columns if c != col]
                )
                return [col, secondary_col]
        return [col]

def display_analysis_results(results):
    """Display analysis results with enhanced formatting."""
    if not results:
        return
    
    # Handle errors
    if "error" in results and results["error"]:
        st.error(f"Error during analysis: {results['error']}")
        return
    
    # Create a clean container for results
    result_container = st.container()
    
    with result_container:
        # Display the text response in a nicely formatted box
        if "text_response" in results and results["text_response"]:
            # Extract and process the text response, removing code blocks for cleaner display
            response_text = results["text_response"]
            
            # Remove markdown code blocks from display text
            clean_response = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
            
            # Clean up extra newlines
            clean_response = re.sub(r'\n{3,}', '\n\n', clean_response)
            
            # Add styled container for insights
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffffff, #f8f9fa); 
                        border-radius: 16px; padding: 1.5rem; 
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                        margin: 1rem 0 2rem 0; border: 1px solid #e5e7eb;">
                <h2 style="margin-top: 0; color: #1E3A8A; font-size: 1.5rem;">Analysis Insights</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Format the response text with headers
            # Find headings and enhance them
            formatted_response = clean_response
            
            # Enhance headings with blue color and proper spacing
            formatted_response = re.sub(r'(#+)\s*(.*)', r'<h3 style="color: #2563EB; margin-top: 1.5rem; font-size: 1.2rem; font-weight: 600;">\2</h3>', formatted_response)
            
            # Bold important statistical findings
            formatted_response = re.sub(r'\*\*(.*?)\*\*', r'<span style="font-weight: 600; color: #1F2937;">\1</span>', formatted_response)
            
            # Highlight key insights
            formatted_response = re.sub(r'(?i)(key (insight|finding|takeaway)s?:?)(.*?)(\n\n|\n(?=\d+\.)|$)', 
                          r'<div style="background: rgba(37, 99, 235, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;"><span style="font-weight: 600; color: #2563EB;">\1</span>\3</div>\4', 
                          formatted_response, flags=re.DOTALL)
            
            # Convert bullet points to styled list items
            formatted_response = re.sub(r'^\s*[-*]\s*(.*?)$', 
                                      r'<li style="margin-bottom: 0.5rem;">\1</li>', 
                                      formatted_response, flags=re.MULTILINE)
            formatted_response = re.sub(r'(<li.*?</li>\n)+', 
                                      r'<ul style="padding-left: 1.5rem; margin: 1rem 0;">\n\g<0></ul>', 
                                      formatted_response)
            
            # Convert numbered lists to styled list items
            formatted_response = re.sub(r'^\s*(\d+)\.\s*(.*?)$', 
                                      r'<li style="margin-bottom: 0.5rem;">\2</li>', 
                                      formatted_response, flags=re.MULTILINE)
            
            # Add paragraphs and spacing
            formatted_response = re.sub(r'\n\n', r'</p><p style="margin: 1rem 0; line-height: 1.7;">', formatted_response)
            formatted_response = f'<p style="margin: 1rem 0; line-height: 1.7;">{formatted_response}</p>'
            
            st.markdown(formatted_response, unsafe_allow_html=True)
        
        # Display visualizations in a gallery format
        if "visualizations" in results and results["visualizations"]:
            visualizations = results["visualizations"]
            
            if visualizations:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ffffff, #f8f9fa); 
                            border-radius: 16px; padding: 1.5rem; 
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                            margin: 1rem 0 1.5rem 0; border: 1px solid #e5e7eb;">
                    <h2 style="margin-top: 0; color: #1E3A8A; font-size: 1.5rem;">Visualizations</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Determine layout based on number of visualizations
                if len(visualizations) == 1:
                    # Single visualization - full width
                    for viz in visualizations:
                        if viz["type"] == "plotly":
                            st.components.v1.html(open(viz["path"], 'r').read(), height=600)
                        else:
                            st.image(viz["path"], use_column_width=True)
                
                elif len(visualizations) == 2:
                    # Two visualizations - side by side
                    cols = st.columns(2)
                    for i, viz in enumerate(visualizations):
                        with cols[i % 2]:
                            if viz["type"] == "plotly":
                                st.components.v1.html(open(viz["path"], 'r').read(), height=400)
                            else:
                                st.image(viz["path"], use_column_width=True)
                
                else:
                    # Multiple visualizations - grid layout
                    viz_per_row = 2
                    for i in range(0, len(visualizations), viz_per_row):
                        cols = st.columns(viz_per_row)
                        for j in range(viz_per_row):
                            if i + j < len(visualizations):
                                with cols[j]:
                                    viz = visualizations[i + j]
                                    if viz["type"] == "plotly":
                                        st.components.v1.html(open(viz["path"], 'r').read(), height=350)
                                    else:
                                        st.image(viz["path"], use_column_width=True)
        
        # Display generated tables
        if "tables" in results and results["tables"]:
            tables = results["tables"]
            
            if tables:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ffffff, #f8f9fa); 
                            border-radius: 16px; padding: 1.5rem; 
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                            margin: 1rem 0 1.5rem 0; border: 1px solid #e5e7eb;">
                    <h2 style="margin-top: 0; color: #1E3A8A; font-size: 1.5rem;">Data Tables</h2>
                </div>
                """, unsafe_allow_html=True)
                
                for table in tables:
                    with st.expander(f"**Table: {table['name']}**", expanded=True):
                        # Style the dataframe
                        styled_df = table["data"].style.format(precision=2)
                        styled_df = styled_df.set_properties(**{
                            'font-family': 'Inter, sans-serif',
                            'border': '1px solid #E5E7EB',
                            'padding': '0.5rem',
                            'text-align': 'left'
                        })
                        
                        st.dataframe(styled_df, use_container_width=True)
        
        # Show executed code if enabled
        if st.session_state.get("show_code", True) and "executed_code" in results and results["executed_code"]:
            with st.expander("View Generated Code", expanded=False):
                for i, code_block in enumerate(results["executed_code"]):
                    st.code(code_block, language="python")

def main():
    """Main application function."""
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up the page config
    setup_page_config()
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Create a clean, minimal header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 800; background: linear-gradient(120deg, #1E3A8A, #2563EB); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            AI Data Analyst
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Set up app settings in a more compact way
    col1, col2 = st.columns(2)
    with col1:
        # Get API key
        api_key = display_api_key_input()
    
    with col2:
        # Model selection
        model = display_model_selection()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type="csv",
        help="Upload your data in CSV format",
        key="file_uploader"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            
            # Store in session state
            st.session_state.df = df
            
            # Initialize the analyst
            if model and api_key:
                st.session_state.analyst = DataAnalyst(df, model=model)
                
                # Display a more compact data preview
                display_data_preview(df)
                
                # Display the analysis interface
                display_analysis_interface(df)
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            logger.error(f"File upload error: {str(e)}")
    else:
        display_welcome()

if __name__ == "__main__":
    main() 