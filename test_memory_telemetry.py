#!/usr/bin/env python
"""
Test script for memory and telemetry functionality
"""

import datetime
import sys
import json
from typing import Dict, Any

# Import the memory and telemetry modules
try:
    import cogitarelink_dspy.memory as memory
    import cogitarelink_dspy.telemetry as telemetry
    from cogitarelink.core.graph import GraphManager
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def header(text):
    """Print a section header."""
    print(f"\n{'='*80}\n{text}\n{'='*80}")

def test_memory():
    """Test memory functionality."""
    header("Testing ReflectionStore")
    
    # Create a graph manager
    graph = GraphManager()
    
    # Create a reflection store
    store = memory.ReflectionStore(graph)
    
    # Add a reflection
    reflection_text = "Test reflection from test script"
    tags = ["test", "memory", "json-ld"]
    
    try:
        note_id = store.add(reflection_text, tags)
        print(f"✅ Successfully added reflection with ID: {note_id}")
        
        # Check graph size
        size = graph.size()
        print(f"Graph size: {size} triples")
        
        # Try to retrieve the reflection
        notes = store.retrieve(limit=1)
        print(f"Retrieved {len(notes)} notes")
        
        if notes:
            print(f"Note content: {notes[0].content}")
        
        # Get as prompt
        prompt = store.as_prompt(limit=1)
        print(f"Prompt: {prompt}")
        
        # Dump all triples in the graph for debugging
        print("\nAll triples in graph:")
        all_triples = graph.query()
        for i, (s, p, o) in enumerate(all_triples):
            print(f"{i+1}: {s} | {p} | {o}")
    
    except Exception as e:
        print(f"❌ Error in memory test: {e}")

def test_telemetry():
    """Test telemetry functionality."""
    header("Testing TelemetryStore")
    
    # Create a graph manager
    graph = GraphManager()
    
    # Create a telemetry store
    store = telemetry.TelemetryStore(graph)
    
    # Log a telemetry event
    event_type = "test"
    value = 123
    tool_iri = "urn:test:tool"
    
    try:
        event_id = store.log(event_type, value, tool_iri, 
                            statusCode=200, 
                            additionalInfo="Test telemetry event")
        print(f"✅ Successfully logged telemetry event with ID: {event_id}")
        
        # Check graph size
        size = graph.size()
        print(f"Graph size: {size} triples")
        
        # Dump all triples in the graph for debugging
        print("\nAll triples in graph:")
        all_triples = graph.query()
        for i, (s, p, o) in enumerate(all_triples):
            print(f"{i+1}: {s} | {p} | {o}")
    
    except Exception as e:
        print(f"❌ Error in telemetry test: {e}")

if __name__ == "__main__":
    test_memory()
    test_telemetry()