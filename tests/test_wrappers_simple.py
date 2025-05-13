"""Simplified tests for the wrappers module."""

import pytest
from cogitarelink_dspy.components import COMPONENTS
from cogitarelink_dspy.wrappers import (
    parse_signature, 
    make_tool_wrappers,
    get_tools,
    group_tools_by_layer
)

def test_signature_parser():
    """Test the signature parser correctly extracts parameters."""
    params, return_type = parse_signature("foo(a:str, b:int)")
    assert params == [("a", "str"), ("b", "int")]
    assert return_type is None
    
    params, return_type = parse_signature("foo(a:str, b:int) -> bool")
    assert params == [("a", "str"), ("b", "int")]
    assert return_type == "bool"

def test_wrapper_generator():
    """Test the wrapper generator without initializing all tools."""
    # Create a minimal test registry
    test_registry = {
        "TestTool": {
            "layer": "Test",
            "tool": "TestUtil",
            "doc": "A test utility tool.",
            "calls": "run(param1:str, param2:int) -> str"
        }
    }
    
    # Generate tool wrappers from the test registry
    tools = make_tool_wrappers(test_registry)
    
    # Verify basic properties
    assert len(tools) == 1
    test_tool = tools[0]
    assert test_tool.__name__ == "TestUtil"
    assert "A test utility tool." in test_tool.__doc__
    assert "Layer: Test" in test_tool.__doc__

def test_get_tools():
    """Test that get_tools returns a list of tool classes."""
    tools = get_tools()
    assert isinstance(tools, list)
    assert len(tools) == len(COMPONENTS)
    
    # Check that each tool has the expected attributes
    for tool in tools:
        assert hasattr(tool, '__name__')
        assert hasattr(tool, '__doc__')
        assert hasattr(tool, 'signature')

def test_group_tools_by_layer():
    """Test grouping tools by their semantic layer."""
    layers_dict = group_tools_by_layer()
    
    # Check expected layers are present
    assert "Context" in layers_dict
    assert "Ontology" in layers_dict
    assert "Rules" in layers_dict
    assert "Instances" in layers_dict
    assert "Verification" in layers_dict
    
    # Check all tools are accounted for
    all_tools = []
    for tools in layers_dict.values():
        all_tools.extend(tools)
    
    assert len(all_tools) == len(COMPONENTS)