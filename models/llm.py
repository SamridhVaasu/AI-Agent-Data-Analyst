import os
import time
import logging
from pandasai.llm.base import LLM
from groq import Groq
import streamlit as st

logger = logging.getLogger("DataAnalysisApp")

class GroqLLM(LLM):
    """
    Custom LLM implementation for Groq API integration.
    Extends the base LLM class from PandasAI to provide specific
    functionality for interacting with Groq's language models.
    """
    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        """
        Initialize the Groq LLM adapter.
        
        Args:
            api_key: Groq API key (will use env variable if not provided)
            model: Model identifier to use for completions
        """
        super().__init__()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.max_retries = 3
        self.retry_delay = 1  # Initial delay in seconds (will exponentially increase)
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Please enter your API key in the text input above.")
            
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Successfully initialized GroqLLM with model: {model}")
        except Exception as e:
            logger.error(f"Error initializing Groq client: {str(e)}")
            raise ValueError(f"Failed to initialize Groq client: {str(e)}")
        
        self.last_prompt = None
    
    def _generate_text(self, prompt, memory=None):
        """
        Generate text from prompt using the Groq API.
        
        Args:
            prompt: The text prompt to send to the model
            memory: Optional memory context (not used in this implementation)
            
        Returns:
            Generated text response from the model
        """
        try:
            self.last_prompt = prompt
            logger.info(f"Sending prompt to Groq API (length: {len(prompt)})")
            
            # Log truncated prompt for debugging
            truncated_prompt = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.info(f"Prompt (truncated): {truncated_prompt}")
            
            # Create completion request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stream=False
            )
            
            result = response.choices[0].message.content
            logger.info(f"Received response from Groq API (length: {len(result)})")
            
            return result
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return f"Sorry, I couldn't generate a response. Error: {str(e)}"
    
    def generate_code(self, instruction, context=None):
        """
        Generate code based on instruction and optional context.
        
        Args:
            instruction: The instruction for code generation
            context: Optional additional context to include
        
        Returns:
            Generated code as a string
        """
        if context:
            prompt = f"{instruction}\n\nContext:\n{context}"
        else:
            prompt = instruction
            
        return self.call(prompt)
    
    def call(self, prompt):
        """
        Call Groq API with retry logic for resilience.
        
        Implements exponential backoff for retries to handle 
        transient API errors gracefully.
        
        Args:
            prompt: The text prompt to send
            
        Returns:
            Model response text
            
        Raises:
            ConnectionError: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=1,
                    stream=False
                )
                return completion.choices[0].message.content
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise ConnectionError(f"Failed to connect to Groq API after {self.max_retries} attempts: {str(e)}")
    
    def is_pandasai_llm(self):
        """Return whether this LLM is a PandasAI LLM."""
        return True
    
    def get_model_name(self):
        """Return the model name."""
        return self.model
    
    def get_temperature(self):
        """Return the temperature used for completions."""
        return 0.7 