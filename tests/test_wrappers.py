"""Tests for the wrappers module."""

import pytest
import dspy
from cogitarelink_dspy.components import COMPONENTS
from cogitarelink_dspy.wrappers import (
    parse_signature,
    make_tool_wrappers,
    group_tools_by_layer,
    TOOLS
)

def test_signature_parser():
    """Test the signature parser correctly extracts parameters."""
    params, return_type = parse_signature("foo(a:str, b:int)")
    assert params == [("a", "str"), ("b", "int")]
    assert return_type is None
    
    params, return_type = parse_signature("foo(a:str, b:int) -> bool")
    assert params == [("a", "str"), ("b", "int")]
    assert return_type == "bool"

def test_wrapper_generation():
    """Test that we generate the correct number of tool wrappers."""
    tools = make_tool_wrappers()
    assert len(tools) == len(COMPONENTS), "Number of tools doesn't match number of components"

def test_tool_properties():
    """Test that generated wrappers have correct properties."""
    # Find the Echo tool
    echo_tool = next(tool for tool in TOOLS if tool.__name__ == "EchoMessage")
    
    # Verify its properties
    assert isinstance(echo_tool.signature, dspy.Signature)
    assert "Utility" in echo_tool.__doc__
    assert "echoes the input message back" in echo_tool.__doc__.lower()
    
    # Instantiate and test basic functionality
    echo = echo_tool()
    assert hasattr(echo, "forward"), "Tool instance should have a forward method"

def test_group_tools_by_layer():
    """Test that tools are properly grouped by layer."""
    layers_dict = group_tools_by_layer()
    
    # Check that we have the expected layers
    expected_layers = ["Context", "Ontology", "Rules", "Instances", "Verification", "Utility"]
    for layer in expected_layers:
        assert layer in layers_dict, f"Missing expected layer: {layer}"
    
    # Check that each tool appears in exactly one layer
    tool_count = 0
    for layer, tools in layers_dict.items():
        tool_count += len(tools)
    
    assert tool_count == len(TOOLS), "Sum of tools in layers doesn't match total tools"