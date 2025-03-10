import os
import streamlit as st

class StreamlitOutput:
    """
    Handler for rendering various types of outputs in the Streamlit UI.
    """
    def __init__(self, container):
        """Initialize with a Streamlit container to render into."""
        self.container = container
        
    def on_code(self, code):
        """Display code with syntax highlighting."""
        self.container.code(code, language="python")
        
    def on_plot(self, plot):
        """Display a plot (image path or plotly figure)."""
        if isinstance(plot, str) and os.path.exists(plot):
            self.container.image(plot)
        else:
            self.container.plotly_chart(plot, use_container_width=True)
        
    def on_dataframe(self, df):
        """Display a pandas DataFrame with Streamlit's data display."""
        self.container.dataframe(df)
        
    def on_error(self, error):
        """Display an error message."""
        self.container.error(f"Error: {error}")
        
    def on_result(self, result):
        """Display a text result."""
        self.container.write(result) 