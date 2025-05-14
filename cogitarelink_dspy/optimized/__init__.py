"""Optimized DSPy models for Cogitarelink.

This module contains pre-trained DSPy models that have been optimized for specific tasks
within the Cogitarelink system. These models are trained using the DSPy compilation framework
and saved for distribution with the package.

Available optimized models:
- memory_planner.pkl: A pre-trained memory planner for handling reflection memory operations
"""

import os
import pickle
from pathlib import Path

def get_optimized_model_path(model_name):
    """Get the path to an optimized model file.
    
    Args:
        model_name (str): The name of the model file (e.g., 'memory_planner.pkl')
        
    Returns:
        str: The full path to the model file
    """
    return os.path.join(os.path.dirname(__file__), model_name)

def model_exists(model_name):
    """Check if an optimized model exists.
    
    Args:
        model_name (str): The name of the model file (e.g., 'memory_planner.pkl')
        
    Returns:
        bool: True if the model exists, False otherwise
    """
    return os.path.exists(get_optimized_model_path(model_name))

def load_model(model_name):
    """Load an optimized model from disk.
    
    Args:
        model_name (str): The name of the model file (e.g., 'memory_planner.pkl')
        
    Returns:
        Any: The loaded model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    model_path = get_optimized_model_path(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Optimized model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)