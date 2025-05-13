#!/usr/bin/env python
"""
Integration test for Cogitarelink DSPy tools.

This script tests the connection between DSPy wrappers and actual Cogitarelink implementations.
"""

import sys
import json
from typing import Dict, Any, List

# DSPy and Cogitarelink imports
try:
    from cogitarelink_dspy.components import COMPONENTS, list_layers, get_tools_by_layer
    from cogitarelink_dspy.wrappers import get_tools, get_tool_by_name, group_tools_by_layer
    from cogitarelink_dspy.core import make_hello_agent
    import cogitarelink.core.context
    import cogitarelink.utils
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

def header(text):
    """Print a section header."""
    print(f"\n{'='*80}\n{text}\n{'='*80}")

def test_component_registry():
    """Test the component registry."""
    header("Testing Component Registry")
    print(f"Found {len(COMPONENTS)} components in the registry")
    layers = list_layers()
    print(f"Available layers: {', '.join(layers)}")
    
    # Check components by layer
    for layer in layers:
        tools = get_tools_by_layer(layer)
        print(f"\nLayer: {layer} - {len(tools)} tools")
        for name, meta in tools.items():
            print(f"  - {name}: {meta['tool']} ({meta['module']})")
            print(f"    Function: {meta['calls']}")
            print(f"    Doc: {meta['doc']}")

def test_wrapper_generation():
    """Test DSPy wrapper generation."""
    header("Testing DSPy Wrapper Generation")
    try:
        tools = get_tools()
        print(f"Successfully generated {len(tools)} tool wrappers")
        
        # Group tools by layer
        layer_groups = group_tools_by_layer(tools)
        for layer, layer_tools in layer_groups.items():
            print(f"\nLayer: {layer} - {len(layer_tools)} tools")
            for tool_class in layer_tools:
                print(f"  - {tool_class.__name__}: {tool_class.module_path}")
                print(f"    Original params: {getattr(tool_class, 'original_params', 'Unknown')}")
                print(f"    Original return: {getattr(tool_class, 'original_return_type', 'Unknown')}")
                print(f"    DSPy signature: {tool_class.signature}")
    
    except Exception as e:
        print(f"Error generating wrappers: {e}")

def test_tool_imports():
    """Test importing actual tools."""
    header("Testing Tool Imports")
    for name, meta in COMPONENTS.items():
        module_path = meta["module"]
        print(f"\nTesting import of {module_path}")
        
        try:
            import importlib
            module = importlib.import_module(module_path)
            print(f"✅ Successfully imported module")
            
            # Try to identify class or function
            method_name = meta['calls'].split('(')[0]
            
            if hasattr(module, method_name):
                print(f"✅ Found function {method_name} directly in module")
            elif hasattr(module, name):
                print(f"✅ Found class {name} in module")
                cls = getattr(module, name)
                instance = cls()
                if hasattr(instance, method_name):
                    print(f"✅ Found method {method_name} in class instance")
                else:
                    print(f"❌ Method {method_name} not found in class instance")
            else:
                # Check if it's in a class with different name
                components = module_path.split('.')
                if len(components) > 1:
                    class_name = components[-1].capitalize()
                    if hasattr(module, class_name):
                        print(f"✅ Found class {class_name} in module")
                        cls = getattr(module, class_name)
                        instance = cls()
                        if hasattr(instance, method_name):
                            print(f"✅ Found method {method_name} in class instance")
                        else:
                            print(f"❌ Method {method_name} not found in class instance")
                    else:
                        print(f"❌ Neither function {method_name} nor class {name}/{class_name} found in module")
                else:
                    print(f"❌ Neither function {method_name} nor class {name} found in module")
            
        except ImportError as e:
            print(f"❌ Failed to import module: {e}")
        except Exception as e:
            print(f"❌ Error checking module: {e}")

def test_simple_tool_call():
    """Test a simple tool call with actual implementation."""
    header("Testing Simple Tool Call with Utils")
    try:
        # Get the LoadModule tool
        tool_class = get_tool_by_name("EchoMessage")
        if not tool_class:
            print("❌ EchoMessage tool not found. Check tool names.")
            return
            
        print(f"Found tool: {tool_class.__name__}")
        tool_instance = tool_class()
        
        # Call the tool with a simple module name
        result = tool_instance(module_name="core", full_name=False)
        print(f"\nResult from tool call:")
        print(f"{result[:200]}..." if len(result) > 200 else result)
        
    except Exception as e:
        print(f"❌ Error calling tool: {e}")
        import traceback
        traceback.print_exc()

def test_context_processor():
    """Test the Context layer with ContextProcessor."""
    header("Testing Context Layer with ContextProcessor")
    try:
        # Get the LoadContext tool
        tool_class = get_tool_by_name("LoadContext")
        if not tool_class:
            print("❌ LoadContext tool not found. Check tool names.")
            return
            
        print(f"Found tool: {tool_class.__name__}")
        tool_instance = tool_class()
        
        # Create a simple document and context
        doc = {"name": "Alice", "knows": "Bob"}
        ctx = {"@context": {"name": "http://schema.org/name", "knows": "http://schema.org/knows"}}
        
        # Call the tool
        try:
            result = tool_instance(doc=doc, ctx=ctx)
            print(f"\nResult from tool call:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"❌ Error in tool execution: {e}")
            
    except Exception as e:
        print(f"❌ Error setting up tool: {e}")
        import traceback
        traceback.print_exc()

def test_hello_agent():
    """Test the HelloAgent with layer detection."""
    header("Testing HelloAgent with Layer Detection")
    try:
        agent = make_hello_agent()
        print("Created HelloAgent")
        
        # Test with messages targeting different layers
        test_messages = [
            "Tell me about JSON-LD contexts", # Context layer
            "How do I use vocabularies in the ontology?", # Ontology layer
            "Can you validate this rule?", # Rules layer
            "How do I store triples in the graph?", # Instances layer
            "Verify the signature on this document", # Verification layer
        ]
        
        for message in test_messages:
            print(f"\nMessage: \"{message}\"")
            result = agent(message)
            print(f"Layer detected: {result.get('layer_used', 'Unknown')}")
            print(f"Tool selected: {result.get('tool_used', 'None')}")
            print(f"Tool result: {result.get('tool_result', 'No result')}")
            
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run all tests
    test_component_registry()
    test_wrapper_generation()
    test_tool_imports()
    test_simple_tool_call()
    test_context_processor()
    test_hello_agent()