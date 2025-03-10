import json
import logging
import traceback
from uuid import uuid4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import linkage, dendrogram
from .llm import GroqLLM

logger = logging.getLogger("DataAnalysisApp")

class DataAnalyst:
    """
    AI-powered data analyst that processes dataframes and generates
    insights and visualizations based on natural language queries.
    """
    def __init__(self, df, model="llama3-70b-8192"):
        """
        Initialize the DataAnalyst with a dataframe and model.
        
        Args:
            df: Pandas DataFrame to analyze
            model: Model identifier to use for LLM completions
        """
        self.df = df
        self.llm = GroqLLM(model=model)
        self.column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        self.summary_stats = self._get_summary_stats()
        logger.info(f"DataAnalyst initialized with model: {model}")
        
    def _get_column_types(self):
        """Get the data types of each column in the dataframe."""
        return self.column_types
    
    def _get_summary_stats(self):
        """
        Generate comprehensive summary statistics for the dataframe.
        
        Includes shape, columns list, numeric stats, categorical stats,
        and missing value counts.
        """
        summary = {}
        # Basic info
        summary["shape"] = self.df.shape
        summary["columns"] = list(self.df.columns)
        
        # Numeric columns stats
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary["numeric_stats"] = self.df[numeric_cols].describe().to_dict()
        
        # Categorical columns stats (limit to first 5 columns for efficiency)
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            summary["categorical_stats"] = {col: self.df[col].value_counts().to_dict() for col in cat_cols[:5]}
            
        # Missing values
        summary["missing_values"] = self.df.isnull().sum().to_dict()
        
        return summary
    
    def generate_basic_visualization(self, column_name):
        """
        Generate a basic visualization for a single column.
        
        Automatically selects appropriate visualization type based on data type:
        - Histogram with KDE for numeric data
        - Bar plot for categorical data
        
        Args:
            column_name: The name of the column to visualize
            
        Returns:
            Path to the saved visualization image
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Choose visualization based on data type
            if pd.api.types.is_numeric_dtype(self.df[column_name]):
                # For numeric data, create a histogram
                sns.histplot(data=self.df, x=column_name, kde=True)
                plt.title(f'Distribution of {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Count')
                
            elif pd.api.types.is_categorical_dtype(self.df[column_name]) or self.df[column_name].nunique() < 10:
                # For categorical data, create a bar plot
                value_counts = self.df[column_name].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'Distribution of {column_name}')
                plt.xticks(rotation=45)
                plt.xlabel(column_name)
                plt.ylabel('Count')
            
            plt.tight_layout()
            
            # Save the figure with a unique filename
            file_id = str(uuid4())[:8]
            output_path = f"viz_{column_name}_{file_id}.png"
            plt.savefig(output_path)
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating visualization for {column_name}: {str(e)}")
            raise ValueError(f"Failed to generate visualization: {str(e)}")
            
    def _create_analysis_prompt(self, query):
        """
        Create a detailed prompt for the LLM that includes context about the 
        dataframe and the user's query.
        
        Args:
            query: User's natural language query about the data
            
        Returns:
            Formatted prompt string with data context
        """
        # Calculate basic dataset properties
        shape_info = f"Dataset Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns"
        columns_info = f"Column Names: {', '.join(self.df.columns)}"
        
        # Get a small sample of the data
        sample_size = min(5, len(self.df))
        sample_df = self.df.head(sample_size)
        
        # Get basic column statistics without detailed value counts
        column_stats = {}
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Use proper function calls for percentiles
                stats = self.df[col].agg({
                    'mean': 'mean',
                    'std': 'std',
                    'min': 'min',
                    'q1': lambda x: x.quantile(0.25),  # 25th percentile
                    'median': 'median',               # 50th percentile
                    'q3': lambda x: x.quantile(0.75),  # 75th percentile
                    'max': 'max'
                }).round(2).to_dict()
                
                # Convert numpy types to Python native types for JSON serialization
                for k, v in stats.items():
                    if hasattr(v, 'item'):
                        stats[k] = v.item()  # Convert numpy type to native Python type
                column_stats[col] = {'type': str(self.df[col].dtype), 'stats': stats}
            else:
                unique_count = self.df[col].nunique()
                # Convert numpy types to Python native types
                if hasattr(unique_count, 'item'):
                    unique_count = unique_count.item()
                column_stats[col] = {'type': str(self.df[col].dtype), 'unique_values': unique_count}

        # Create a concise prompt
        prompt = f"""You are an expert data analyst and visualization specialist. Analyze this dataset which is already loaded into a pandas DataFrame named 'df':

{shape_info}
{columns_info}

Sample Data (first {sample_size} rows):
{sample_df.to_string()}

Column Information:
{json.dumps(column_stats, indent=2)}

User Query: "{query}"

Important Guidelines:
1. The data is already loaded in the 'df' variable - DO NOT try to read any CSV files
2. Use ```python and ``` for code blocks
3. Create highly detailed, publication-quality visualizations using Plotly:
   - import plotly.express as px
   - import plotly.graph_objects as go

Visualization Requirements (make them more detailed):
1. Titles and Captions:
   - Use descriptive titles that fully explain what the visualization shows
   - Add informative subtitles when helpful
   - Include clear axis labels with units when relevant
   - Add annotations to highlight key insights directly on the visualization

2. Data Representation:
   - Use appropriate color scales with high contrast
   - Include error bars or confidence intervals when applicable
   - Show data distributions alongside summary statistics
   - Use size, shape, or color to encode additional dimensions
   - Display exact values on hover

3. Layout and Design:
   - Use a clean white background with minimal gridlines
   - Implement proper spacing and margins
   - Set explicit ranges for axes when needed
   - Customize tick marks and tick labels for readability
   - Use professional font styling (Inter or Arial)

4. Professional Color Palettes:
   - Sequential: ['#002B5B', '#0F52BA', '#2563EB', '#60A5FA', '#93C5FD', '#BFDBFE', '#DBEAFE']
   - Diverging: ['#7F1D1D', '#DC2626', '#F87171', '#FECACA', '#DBEAFE', '#60A5FA', '#2563EB', '#1E3A8A']
   - Categorical: ['#2563EB', '#059669', '#DC2626', '#D97706', '#7C3AED', '#DB2777', '#4338CA', '#A16207']

5. Create Enhanced Layouts:
   - fig.update_layout(
       template='plotly_white',
       font=dict(family='Inter, Arial, sans-serif', size=14),
       title=dict(
           text="<b>Main Title</b><br><sup>Subtitle with additional context</sup>",
           font=dict(size=22, color='#1E3A8A'),
           x=0.5,
           y=0.95
       ),
       legend=dict(
           borderwidth=1,
           bordercolor='#E5E7EB',
           font=dict(size=12)
       ),
       margin=dict(l=80, r=80, t=100, b=80),
       plot_bgcolor='white',
       paper_bgcolor='white',
       xaxis=dict(
           showgrid=True,
           gridcolor='#E5E7EB',
           zeroline=True,
           zerolinecolor='#9CA3AF',
           showline=True,
           linewidth=1,
           linecolor='#6B7280'
       ),
       yaxis=dict(
           showgrid=True,
           gridcolor='#E5E7EB',
           zeroline=True,
           zerolinecolor='#9CA3AF',
           showline=True,
           linewidth=1,
           linecolor='#6B7280'
       ),
       coloraxis_colorbar=dict(
           thickness=20,
           lenmode='fraction',
           len=0.75,
           outlinewidth=1,
           outlinecolor='#E5E7EB'
       )
   )

6. Enhance Each Plot Type:
   a. Histograms:
      - Add kernel density estimate curves: fig.add_trace(go.Scatter(x=x, y=density, mode='lines', name='Density'))
      - Mark mean/median with vertical lines: fig.add_vline(x=mean, line_dash="dash", line_color="#DC2626")
      - Add rug plots with go.Scatter(x=data, y=[0]*len(data), mode='markers', marker=dict(color='black', symbol='line-ns'))
      - Show distribution statistics in annotations

   b. Scatter Plots:
      - Add trendlines with confidence bands using px.scatter(..., trendline="ols", trendline_color_override="red")
      - Highlight outliers with different symbols or colors
      - Add marginal distributions on x and y axes
      - Include correlation coefficient and p-value in annotations

   c. Bar Charts:
      - Sort bars by value for better readability
      - Add value labels on bars: text=values, textposition='auto'
      - Group related categories with facets or subplots
      - Use color gradients to show additional dimensions

   d. Line Charts:
      - Add range bands for uncertainty using go.Scatter(fill='tonexty')
      - Mark significant events with go.Scatter(mode='markers')
      - Use hover templates with multiple metrics
      - Highlight important segments with colors or annotations

   e. Box Plots and Violin Plots:
      - Combine box plots with violin plots for distribution shape
      - Show all data points for small datasets using px.box(..., points='all')
      - Add means as markers inside the boxes
      - Use notches for confidence intervals around medians

   f. Heatmaps and Correlation Matrices:
      - Add dendrograms for clustered heatmaps
      - Display exact values in each cell with texttemplate
      - Use diverging color scales centered at meaningful values
      - Add marginal bar charts for row/column summaries

7. Add Interactive Elements:
   - Create buttons for different views: updatemenus=[dict(buttons=[...])]
   - Add range sliders for time series or large datasets
   - Implement dropdowns for filtering by categories
   - Create tabs for different chart types of the same data

Please provide:
1. A detailed, insightful analysis answering the query
2. Python code that creates highly detailed and professional visualizations
3. Statistical findings with appropriate significance tests when relevant
4. Clear explanations of patterns, trends, and anomalies in the data"""
        return prompt
    
    def _extract_code_blocks(self, text):
        """
        Extract Python code blocks from LLM response text.
        
        Args:
            text: Response text from LLM
            
        Returns:
            List of extracted code blocks
        """
        import re
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]
    
    def generate_visualization(self, viz_type, columns, title=None):
        """
        Generate a visualization of the specified type for the given columns.
        
        Args:
            viz_type: Type of visualization (histogram, scatter, bar, etc.)
            columns: List of columns to visualize
            title: Title for the visualization
            
        Returns:
            Visualization results
        """
        try:
            # Import here to avoid issues
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.stats import gaussian_kde
            
            logger.info(f"Generating {viz_type} visualization for columns: {columns}")
            
            results = {
                "visualizations": [],
                "executed_code": []
            }
            
            # Set default title if not provided
            if not title:
                title = f"{viz_type.capitalize()} of {', '.join(columns)}"
            
            # Basic styling
            layout = dict(
                template="plotly_white",
                title=dict(
                    text=f"<b>{title}</b>",
                    font=dict(size=22, color="#1E3A8A"),
                    x=0.5
                ),
                legend=dict(
                    borderwidth=1,
                    bordercolor="#E5E7EB"
                ),
                margin=dict(l=80, r=80, t=100, b=80),
                font=dict(family="Inter, Arial, sans-serif", size=14),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(
                    title=dict(font=dict(size=16)),
                    showgrid=True,
                    gridcolor="#E5E7EB",
                    zeroline=True,
                    zerolinecolor="#9CA3AF",
                    showline=True,
                    linewidth=1,
                    linecolor="#6B7280"
                ),
                yaxis=dict(
                    title=dict(font=dict(size=16)),
                    showgrid=True,
                    gridcolor="#E5E7EB",
                    zeroline=True,
                    zerolinecolor="#9CA3AF",
                    showline=True,
                    linewidth=1,
                    linecolor="#6B7280"
                ),
                coloraxis_colorbar=dict(
                    thickness=20,
                    outlinewidth=1,
                    outlinecolor="#E5E7EB"
                )
            )
            
            fig = None
            
            # Generate visualization based on type
            if viz_type == "histogram":
                # Handle single column histograms
                if len(columns) == 1:
                    col = columns[0]
                    
                    # Skip if column is not numeric
                    if not pd.api.types.is_numeric_dtype(self.df[col]):
                        return {
                            "error": f"Column {col} is not numeric. Cannot create histogram."
                        }
                    
                    # Create histogram with density curve
                    fig = px.histogram(
                        self.df, 
                        x=col,
                        histnorm="probability density",
                        marginal="box",  # Add box plot on the margin
                        title=title,
                        color_discrete_sequence=["#2563EB"],
                        opacity=0.7,
                        template="plotly_white"
                    )
                    
                    # Add kernel density estimate
                    data = self.df[col].dropna()
                    if len(data) > 1:  # Need at least 2 points for KDE
                        kde = gaussian_kde(data)
                        x_range = np.linspace(data.min(), data.max(), 1000)
                        y_kde = kde(x_range)
                        fig.add_trace(go.Scatter(
                            x=x_range, 
                            y=y_kde, 
                            mode='lines', 
                            name='Density',
                            line=dict(color='#DC2626', width=2)
                        ))
                    
                    # Add mean and median lines
                    mean_val = data.mean()
                    median_val = data.median()
                    fig.add_vline(
                        x=mean_val, 
                        line_dash="dash", 
                        line_color="#DC2626",
                        annotation_text=f"Mean: {mean_val:.2f}",
                        annotation_position="top right"
                    )
                    fig.add_vline(
                        x=median_val, 
                        line_dash="dot", 
                        line_color="#0891B2",
                        annotation_text=f"Median: {median_val:.2f}",
                        annotation_position="bottom right"
                    )
                    
                    # Add statistics annotation
                    std_val = data.std()
                    fig.add_annotation(
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        text=f"<b>Statistics:</b><br>Mean: {mean_val:.2f}<br>Median: {median_val:.2f}<br>Std Dev: {std_val:.2f}",
                        showarrow=False,
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="#E5E7EB",
                        borderwidth=1,
                        borderpad=4,
                        font=dict(size=12)
                    )
                    
                    # Update layout with enhanced styling
                    fig.update_layout(layout)
                    fig.update_layout(
                        xaxis_title=col,
                        yaxis_title="Density"
                    )
                    
                # Multiple columns get separate histograms in subplots
                else:
                    # Filter to only numeric columns
                    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])]
                    
                    if not numeric_cols:
                        return {
                            "error": "None of the selected columns are numeric. Cannot create histograms."
                        }
                    
                    # Create subplot grid
                    from plotly.subplots import make_subplots
                    rows = (len(numeric_cols) + 1) // 2  # Ceiling division
                    cols = min(2, len(numeric_cols))
                    
                    fig = make_subplots(
                        rows=rows, 
                        cols=cols, 
                        subplot_titles=[f"{col} Distribution" for col in numeric_cols]
                    )
                    
                    # Add histograms for each column
                    for i, col in enumerate(numeric_cols):
                        row = i // 2 + 1
                        col_idx = i % 2 + 1
                        
                        data = self.df[col].dropna()
                        
                        # Add histogram
                        fig.add_trace(
                            go.Histogram(
                                x=data,
                                name=col,
                                marker=dict(
                                    color="#2563EB",
                                    opacity=0.7,
                                    line=dict(color="white", width=0.5)
                                ),
                                showlegend=False
                            ),
                            row=row, col=col_idx
                        )
                        
                        # Add KDE if enough data points
                        if len(data) > 1:
                            kde = gaussian_kde(data)
                            x_range = np.linspace(data.min(), data.max(), 1000)
                            y_kde = kde(x_range)
                            fig.add_trace(
                                go.Scatter(
                                    x=x_range, 
                                    y=y_kde * (data.count() / y_kde.max()),  # Scale to match histogram height
                                    mode='lines', 
                                    name='Density',
                                    line=dict(color='#DC2626', width=2),
                                    showlegend=False
                                ),
                                row=row, col=col_idx
                            )
                        
                        # Add mean line
                        mean_val = data.mean()
                        fig.add_vline(
                            x=mean_val, 
                            line_dash="dash", 
                            line_color="#DC2626",
                            row=row, col=col_idx
                        )
                    
                    # Update layout with enhanced styling
                    fig.update_layout(
                        title=dict(
                            text=f"<b>{title}</b>",
                            font=dict(size=22, color="#1E3A8A"),
                            x=0.5
                        ),
                        template="plotly_white",
                        showlegend=False,
                        margin=dict(l=60, r=60, t=100, b=60),
                        font=dict(family="Inter, Arial, sans-serif"),
                        plot_bgcolor="white",
                        paper_bgcolor="white"
                    )
                    
            elif viz_type == "scatter":
                # Scatter plots require exactly two numeric columns
                if len(columns) != 2:
                    return {
                        "error": "Scatter plots require exactly two columns."
                    }
                
                col_x, col_y = columns
                
                # Verify columns are numeric
                if not (pd.api.types.is_numeric_dtype(self.df[col_x]) and 
                        pd.api.types.is_numeric_dtype(self.df[col_y])):
                    return {
                        "error": f"Both columns must be numeric for scatter plots."
                    }
                
                # Find a potential color column (categorical with few unique values)
                color_col = None
                for col in self.df.columns:
                    if col not in columns and self.df[col].nunique() <= 10:
                        color_col = col
                        break
                
                # Create enhanced scatter plot
                fig = px.scatter(
                    self.df,
                    x=col_x,
                    y=col_y,
                    color=color_col,  # Use color column if found
                    color_discrete_sequence=["#2563EB", "#059669", "#DC2626", "#D97706", "#7C3AED", "#DB2777"],
                    opacity=0.7,
                    marginal_x="histogram",  # Add histograms on margins
                    marginal_y="histogram",
                    trendline="ols",  # Add trendline
                    trendline_color_override="#DC2626",
                    title=title,
                    template="plotly_white",
                    hover_data=self.df.columns[:5]  # Include additional columns in hover
                )
                
                # Calculate correlation
                corr = self.df[[col_x, col_y]].corr().iloc[0,1]
                
                # Add correlation annotation
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    text=f"<b>Correlation:</b> {corr:.3f}",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#E5E7EB",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=12)
                )
                
                # Update layout with enhanced styling
                fig.update_layout(layout)
                fig.update_layout(
                    xaxis_title=col_x,
                    yaxis_title=col_y
                )
                
            elif viz_type == "bar":
                # Bar charts work best with one categorical and one numeric column
                if len(columns) < 1 or len(columns) > 2:
                    return {
                        "error": "Bar charts require one or two columns."
                    }
                
                # One column: frequency count
                if len(columns) == 1:
                    col = columns[0]
                    value_counts = self.df[col].value_counts().sort_values(ascending=False)
                    
                    # Limit to top 20 categories if there are too many
                    if len(value_counts) > 20:
                        value_counts = value_counts.head(20)
                        title = f"Top 20 {col} Counts"
                    
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        color=value_counts.values,
                        color_continuous_scale=["#BFDBFE", "#60A5FA", "#2563EB", "#1E40AF", "#1E3A8A"],
                        title=title,
                        template="plotly_white",
                        labels={"x": col, "y": "Count", "color": "Count"}
                    )
                    
                    # Add value labels on bars
                    fig.update_traces(
                        texttemplate='%{y}',
                        textposition='outside'
                    )
                    
                    # Add percentage annotation
                    total = sum(value_counts.values)
                    fig.add_annotation(
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        text=f"<b>Total:</b> {total}",
                        showarrow=False,
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="#E5E7EB",
                        borderwidth=1,
                        borderpad=4,
                        font=dict(size=12)
                    )
                    
                # Two columns: groupby aggregation
                else:
                    cat_col, val_col = None, None
                    
                    # Determine which column is categorical and which is numeric
                    if pd.api.types.is_numeric_dtype(self.df[columns[0]]):
                        if not pd.api.types.is_numeric_dtype(self.df[columns[1]]):
                            val_col, cat_col = columns[0], columns[1]
                        else:
                            # Both numeric, use the one with fewer unique values as categorical
                            if self.df[columns[0]].nunique() < self.df[columns[1]].nunique():
                                cat_col, val_col = columns[0], columns[1]
                            else:
                                cat_col, val_col = columns[1], columns[0]
                    else:
                        if pd.api.types.is_numeric_dtype(self.df[columns[1]]):
                            cat_col, val_col = columns[0], columns[1]
                        else:
                            # Both categorical, can't create bar chart with aggregation
                            return {
                                "error": "At least one column must be numeric for aggregated bar charts."
                            }
                    
                    # Limit categories if there are too many
                    categories = self.df[cat_col].value_counts().sort_values(ascending=False)
                    if len(categories) > 20:
                        top_categories = categories.head(20).index
                        filtered_df = self.df[self.df[cat_col].isin(top_categories)].copy()
                        title = f"Average {val_col} by Top 20 {cat_col} Categories"
                    else:
                        filtered_df = self.df.copy()
                    
                    # Create aggregated bar chart
                    agg_data = filtered_df.groupby(cat_col)[val_col].agg(['mean', 'count']).reset_index()
                    agg_data = agg_data.sort_values('mean', ascending=False)
                    
                    fig = px.bar(
                        agg_data,
                        x=cat_col,
                        y='mean',
                        color='mean',
                        color_continuous_scale=["#BFDBFE", "#60A5FA", "#2563EB", "#1E40AF", "#1E3A8A"],
                        title=title,
                        template="plotly_white",
                        labels={cat_col: cat_col, "mean": f"Average {val_col}", "count": "Count"},
                        hover_data=['count']
                    )
                    
                    # Add value labels on bars
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='outside'
                    )
                
                # Update layout with enhanced styling
                fig.update_layout(layout)
                
            elif viz_type == "line":
                # Line charts work best with a time series or ordered data
                if len(columns) < 1 or len(columns) > 3:
                    return {
                        "error": "Line charts require 1-3 columns (x-axis, y-axis, and optional color)."
                    }
                
                # Try to find appropriate x and y columns
                x_col, y_col, color_col = None, None, None
                
                if len(columns) == 1:
                    # Single column: use index as x and column as y
                    y_col = columns[0]
                    
                    # If column is not numeric, count values over index
                    if not pd.api.types.is_numeric_dtype(self.df[y_col]):
                        return {
                            "error": f"Column {y_col} is not numeric. Cannot create line chart."
                        }
                    
                    fig = px.line(
                        self.df.reset_index(),
                        x='index',
                        y=y_col,
                        title=title,
                        template="plotly_white",
                        line_shape="spline",  # Smooth curve
                        render_mode="svg",    # Cleaner lines
                        color_discrete_sequence=["#2563EB"]
                    )
                    
                    # Add markers for data points
                    fig.update_traces(mode="lines+markers", marker=dict(size=6))
                    
                    # Update layout with enhanced styling
                    fig.update_layout(layout)
                    fig.update_layout(
                        xaxis_title="Index",
                        yaxis_title=y_col
                    )
                    
                elif len(columns) >= 2:
                    # Try to determine appropriate columns
                    # If one column is datetime, use it as x
                    datetime_cols = [col for col in columns if pd.api.types.is_datetime64_dtype(self.df[col])]
                    numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])]
                    
                    if datetime_cols:
                        x_col = datetime_cols[0]
                        numeric_cols = [c for c in numeric_cols if c != x_col]
                        if numeric_cols:
                            y_col = numeric_cols[0]
                        else:
                            return {
                                "error": "Need at least one numeric column for line chart y-axis."
                            }
                    elif numeric_cols:
                        if len(numeric_cols) >= 2:
                            # Use column with more unique values as x
                            if self.df[numeric_cols[0]].nunique() > self.df[numeric_cols[1]].nunique():
                                x_col, y_col = numeric_cols[0], numeric_cols[1]
                            else:
                                x_col, y_col = numeric_cols[1], numeric_cols[0]
                        else:
                            # Only one numeric column, use index as x
                            y_col = numeric_cols[0]
                            x_col = 'index'
                            self.df = self.df.reset_index()
                    else:
                        return {
                            "error": "Need at least one numeric column for line chart."
                        }
                    
                    # Optional color column
                    if len(columns) == 3:
                        remaining_cols = [c for c in columns if c != x_col and c != y_col]
                        if remaining_cols:
                            color_col = remaining_cols[0]
                    
                    # Create enhanced line chart
                    fig = px.line(
                        self.df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=title,
                        template="plotly_white",
                        line_shape="spline",  # Smooth curve
                        render_mode="svg",    # Cleaner lines
                        color_discrete_sequence=["#2563EB", "#059669", "#DC2626", "#D97706", "#7C3AED"]
                    )
                    
                    # Add markers for data points
                    fig.update_traces(mode="lines+markers", marker=dict(size=6))
                    
                    # Add range band for uncertainty if no color column
                    if not color_col and pd.api.types.is_numeric_dtype(self.df[y_col]):
                        # Calculate rolling mean and standard deviation
                        window = max(5, len(self.df) // 20)  # Adaptive window size
                        rolling_mean = self.df[y_col].rolling(window=window, center=True).mean()
                        rolling_std = self.df[y_col].rolling(window=window, center=True).std()
                        
                        # Add range band
                        fig.add_trace(
                            go.Scatter(
                                x=self.df[x_col],
                                y=rolling_mean + rolling_std,
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=self.df[x_col],
                                y=rolling_mean - rolling_std,
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(37, 99, 235, 0.2)',
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )
                    
                    # Update layout with enhanced styling
                    fig.update_layout(layout)
                    fig.update_layout(
                        xaxis_title=x_col,
                        yaxis_title=y_col
                    )
                
            elif viz_type == "box":
                # Box plots work with one or more columns
                numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])]
                
                if not numeric_cols:
                    return {
                        "error": "No numeric columns selected. Cannot create box plot."
                    }
                
                # Check if we have a potential grouping column
                group_col = None
                for col in columns:
                    if col not in numeric_cols and self.df[col].nunique() <= 10:
                        group_col = col
                        break
                
                if group_col:
                    # Box plot with grouping
                    value_cols = [c for c in numeric_cols if c != group_col]
                    
                    if not value_cols:
                        return {
                            "error": "Need at least one numeric column for box plot values."
                        }
                    
                    # Melt dataframe for multiple value columns
                    if len(value_cols) > 1:
                        melted_df = pd.melt(
                            self.df,
                            id_vars=[group_col],
                            value_vars=value_cols,
                            var_name="Variable",
                            value_name="Value"
                        )
                        
                        fig = px.box(
                            melted_df,
                            x=group_col,
                            y="Value",
                            color="Variable",
                            title=title,
                            template="plotly_white",
                            color_discrete_sequence=["#2563EB", "#059669", "#DC2626", "#D97706", "#7C3AED"],
                            points="all",  # Show all points
                            notched=True   # Show confidence intervals
                        )
                    else:
                        # Single value column
                        fig = px.box(
                            self.df,
                            x=group_col,
                            y=value_cols[0],
                            title=title,
                            template="plotly_white",
                            color=group_col,
                            color_discrete_sequence=["#2563EB", "#059669", "#DC2626", "#D97706", "#7C3AED"],
                            points="all",  # Show all points
                            notched=True   # Show confidence intervals
                        )
                else:
                    # Simple box plot for multiple columns
                    fig = px.box(
                        self.df,
                        y=numeric_cols,
                        title=title,
                        template="plotly_white",
                        color_discrete_sequence=["#2563EB", "#059669", "#DC2626", "#D97706", "#7C3AED"],
                        points="all",  # Show all points
                        notched=True   # Show confidence intervals
                    )
                    
                    # Add mean markers
                    for i, col in enumerate(numeric_cols):
                        mean_val = self.df[col].mean()
                        fig.add_trace(
                            go.Scatter(
                                x=[i],
                                y=[mean_val],
                                mode="markers",
                                marker=dict(
                                    color="red",
                                    size=10,
                                    symbol="diamond"
                                ),
                                name=f"Mean of {col}",
                                showlegend=False
                            )
                        )
                
                # Update layout with enhanced styling
                fig.update_layout(layout)
                
            elif viz_type == "heatmap":
                # Heatmaps work best with numeric columns for correlation
                numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(self.df[col])]
                
                if len(numeric_cols) < 2:
                    return {
                        "error": "Need at least two numeric columns for heatmap."
                    }
                
                # Create correlation matrix
                corr_matrix = self.df[numeric_cols].corr()
                
                # Create enhanced heatmap
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                # Create heatmap with dendrograms for clustering
                dendro_leaves = None
                if len(numeric_cols) > 2:
                    try:
                        # Compute linkage for clustering
                        link = linkage(corr_matrix, 'ward')
                        dendro = dendrogram(link, no_plot=True)
                        dendro_leaves = dendro['leaves']
                        
                        # Reorder correlation matrix
                        corr_matrix = corr_matrix.iloc[dendro_leaves, dendro_leaves]
                    except:
                        # Skip clustering if there's an error
                        pass
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title=title,
                    template="plotly_white"
                )
                
                # Add correlation values as text
                fig.update_traces(
                    text=corr_matrix.round(2).values,
                    texttemplate="%{text:.2f}"
                )
                
                # Highlight strong correlations
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        if i != j and abs(corr_matrix.iloc[i, j]) > 0.7:
                            fig.add_annotation(
                                x=j,
                                y=i,
                                text="!",
                                showarrow=False,
                                font=dict(
                                    color="black",
                                    size=12
                                )
                            )
                
                # Update layout with enhanced styling
                fig.update_layout(layout)
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title=""
                )
            
            # Create visualization output
            if fig:
                # Set figure size
                fig.update_layout(
                    width=900,
                    height=600
                )
                
                # Save to HTML
                fig_id = f"fig_{uuid4()}"
                fig_path = f"{fig_id}.html"
                fig.write_html(fig_path)
                
                # Add to results
                results["visualizations"].append({
                    "id": fig_id,
                    "path": fig_path,
                    "type": "plotly"
                })
                
                # Generate representative code
                import inspect
                code = inspect.getsource(self.generate_visualization)
                results["executed_code"].append(code)
                
                return results
            else:
                return {
                    "error": f"Could not generate visualization for type {viz_type} with columns {columns}"
                }
        
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": f"Error generating visualization: {str(e)}"
            }

    def analyze(self, query):
        """
        Analyze the dataframe based on a natural language query.
        
        Uses the LLM to interpret the query, generate code for analysis,
        and produce visualizations and insights.
        
        Args:
            query: Natural language query about the data
            
        Returns:
            Analysis results including text, code, and visualizations
        """
        try:
            logger.info(f"Starting analysis for query: {query}")
            
            # Create the initial prompt
            prompt = self._create_analysis_prompt(query)
            logger.info("Created analysis prompt")
            
            # Get response from LLM
            try:
                response = self.llm.call(prompt)
                logger.info("Received response from LLM")
            except Exception as e:
                logger.error(f"LLM error: {str(e)}")
                
                # Try to extract a column name from the query
                column_matches = [col for col in self.df.columns 
                                 if col.lower() in query.lower()]
                
                if column_matches:
                    logger.info(f"Falling back to basic visualization for column: {column_matches[0]}")
                    return self.generate_basic_visualization(column_matches[0])
                else:
                    return {
                        "text_response": f"Error connecting to LLM: {str(e)}. Please try again later.",
                        "visualizations": [],
                        "tables": [],
                        "executed_code": [],
                        "error": str(e)
                    }
            
            # Process the response
            results = {
                "text_response": response,
                "visualizations": [],
                "tables": [],
                "executed_code": []
            }
            
            # Extract and execute code blocks
            code_blocks = self._extract_code_blocks(response)
            if code_blocks:
                logger.info(f"Found {len(code_blocks)} code blocks to execute")
                for i, code in enumerate(code_blocks):
                    try:
                        logger.info(f"Executing code block {i+1}")
                        
                        # Add code to results
                        results["executed_code"].append(code)
                        
                        # Create execution environment with plotly
                        local_vars = {
                            "df": self.df,
                            "pd": pd,
                            "plt": plt,
                            "sns": sns,
                            "np": np,
                            "px": px,
                            "go": go,
                            "pio": pio
                        }
                        
                        # Set a nice template for all plotly figures
                        pio.templates.default = "plotly_white"
                        
                        # Execute the code
                        exec(code, globals(), local_vars)
                        
                        # Handle Plotly figures
                        for var_name, var_value in local_vars.items():
                            if isinstance(var_value, (go.Figure, px.Figure)) and var_name != "fig":
                                # Save Plotly figure to HTML
                                fig_id = f"fig_{uuid4()}"
                                fig_path = f"{fig_id}.html"
                                var_value.write_html(fig_path)
                                results["visualizations"].append({
                                    "id": fig_id,
                                    "path": fig_path,
                                    "code": code,
                                    "type": "plotly"
                                })
                                logger.info(f"Saved Plotly figure as HTML: {fig_path}")
                        
                        # Handle matplotlib figures if present
                        if plt.get_fignums():
                            plt.tight_layout()
                            fig_id = f"fig_{uuid4()}"
                            fig_path = f"{fig_id}.png"
                            plt.savefig(fig_path, bbox_inches='tight', dpi=300)
                            plt.close()
                            results["visualizations"].append({
                                "id": fig_id,
                                "path": fig_path,
                                "code": code,
                                "type": "matplotlib"
                            })
                            logger.info(f"Created matplotlib visualization: {fig_path}")
                            
                        # Handle generated dataframes
                        for var_name, var_value in local_vars.items():
                            if isinstance(var_value, pd.DataFrame) and var_name != "df":
                                # Limit output size for large dataframes
                                if len(var_value) > 100:
                                    var_value = var_value.head(100)
                                results["tables"].append({"name": var_name, "data": var_value})
                        
                    except Exception as e:
                        error_msg = f"Error executing code block {i}: {str(e)}"
                        logger.error(error_msg)
                        logger.error(f"Problematic code:\n{code}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        results["error"] = error_msg
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "text_response": f"An error occurred during analysis: {str(e)}",
                "visualizations": [],
                "tables": [],
                "executed_code": [],
                "error": str(e)
            } 