#!/usr/bin/env python
"""
Test script for DSPy wrappers
"""

import sys
import json
from typing import Dict, Any

# Import the wrappers
try:
    from cogitarelink_dspy.wrappers import get_tool_by_name
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def header(text):
    """Print a section header."""
    print(f"\n{'='*80}\n{text}\n{'='*80}")

def test_memory_wrappers():
    """Test memory tool wrappers."""
    header("Testing Memory Tool Wrappers")
    
    try:
        # Get the AddReflection tool
        add_tool = get_tool_by_name("AddReflection")
        if not add_tool:
            print("❌ AddReflection tool not found")
            return
            
        print(f"Found tool: {add_tool.__name__}")
        add_instance = add_tool()
        
        # Add a reflection
        add_result = add_instance(text="Test reflection via DSPy wrapper", tags=["dspy", "wrapper"])
        print(f"✅ Added reflection: {add_result}")
        
        # Get the RecallReflection tool
        recall_tool = get_tool_by_name("RecallReflection")
        if not recall_tool:
            print("❌ RecallReflection tool not found")
            return
            
        print(f"Found tool: {recall_tool.__name__}")
        recall_instance = recall_tool()
        
        # Recall reflections
        recall_result = recall_instance(limit=5)
        print(f"✅ Recalled reflections: {recall_result}")
        
        # Get the ReflectionPrompt tool
        prompt_tool = get_tool_by_name("ReflectionPrompt")
        if not prompt_tool:
            print("❌ ReflectionPrompt tool not found")
            return
            
        print(f"Found tool: {prompt_tool.__name__}")
        prompt_instance = prompt_tool()
        
        # Generate prompt
        prompt_result = prompt_instance(limit=5)
        print(f"✅ Generated prompt: {prompt_result}")
        
    except Exception as e:
        print(f"❌ Error in memory wrapper test: {e}")
        import traceback
        traceback.print_exc()

def test_telemetry_wrapper():
    """Test telemetry tool wrapper."""
    header("Testing Telemetry Tool Wrapper")
    
    try:
        # Get the LogTelemetry tool
        log_tool = get_tool_by_name("LogTelemetry")
        if not log_tool:
            print("❌ LogTelemetry tool not found")
            return
            
        print(f"Found tool: {log_tool.__name__}")
        log_instance = log_tool()
        
        # Log a telemetry event
        log_result = log_instance(
            event_type="test_wrapper", 
            value=456, 
            tool_iri="urn:test:dspy:wrapper",
            statusCode=201
        )
        print(f"✅ Logged telemetry event: {log_result}")
        
    except Exception as e:
        print(f"❌ Error in telemetry wrapper test: {e}")
        import traceback
        traceback.print_exc()

def test_context_wrapper():
    """Test context tool wrapper."""
    header("Testing Context Tool Wrapper")
    
    try:
        # Get the LoadContext tool
        load_tool = get_tool_by_name("LoadContext")
        if not load_tool:
            print("❌ LoadContext tool not found")
            return
            
        print(f"Found tool: {load_tool.__name__}")
        load_instance = load_tool()
        
        # Create a simple document and context
        doc = {"name": "Alice", "knows": "Bob"}
        ctx = {"@context": {"name": "http://schema.org/name", "knows": "http://schema.org/knows"}}
        
        # Call the tool
        load_result = load_instance(doc=doc, ctx=ctx)
        print(f"✅ Result from LoadContext:")
        print(json.dumps(load_result, indent=2))
        
    except Exception as e:
        print(f"❌ Error in context wrapper test: {e}")
        import traceback
        traceback.print_exc()

def test_utils_wrapper():
    """Test utils tool wrapper."""
    header("Testing Utils Tool Wrapper")
    
    try:
        # Get the EchoMessage tool
        echo_tool = get_tool_by_name("EchoMessage")
        if not echo_tool:
            print("❌ EchoMessage tool not found")
            return
            
        print(f"Found tool: {echo_tool.__name__}")
        echo_instance = echo_tool()
        
        # Call the tool
        echo_result = echo_instance(module_name="utils", full_name=False)
        print(f"✅ Result from EchoMessage:")
        print(f"{echo_result[:200]}..." if len(echo_result) > 200 else echo_result)
        
    except Exception as e:
        print(f"❌ Error in utils wrapper test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_wrappers()
    test_telemetry_wrapper()
    test_context_wrapper()
    test_utils_wrapper()