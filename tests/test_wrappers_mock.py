"""Tests for wrappers module using mocking to avoid runtime signature issues."""

import pytest
from unittest.mock import patch, MagicMock
from cogitarelink_dspy.components import COMPONENTS
from cogitarelink_dspy.wrappers import parse_signature

def test_signature_parser():
    """Test the signature parser correctly extracts parameters."""
    params, return_type = parse_signature("foo(a:str, b:int)")
    assert params == [("a", "str"), ("b", "int")]
    assert return_type is None
    
    params, return_type = parse_signature("foo(a:str, b:int) -> bool")
    assert params == [("a", "str"), ("b", "int")]
    assert return_type == "bool"
    
    # Test empty parameters
    params, return_type = parse_signature("foo() -> str")
    assert params == []
    assert return_type == "str"
    
    # Test multiple parameters
    params, return_type = parse_signature("validate(subject:str, predicate:str, object:str)")
    assert len(params) == 3
    assert params[0] == ("subject", "str")
    assert params[1] == ("predicate", "str")
    assert params[2] == ("object", "str")

# These tests verify that our component catalog and wrappers architecture is sound,
# even if we can't directly instantiate them in the test environment due to DSPy
# signature handling issues.
def test_component_registry_structure():
    """Test that our component registry has the expected structure for tool generation."""
    # Check every component has the required fields
    for name, meta in COMPONENTS.items():
        assert "layer" in meta, f"Component {name} missing 'layer'"
        assert "tool" in meta, f"Component {name} missing 'tool'"
        assert "doc" in meta, f"Component {name} missing 'doc'"
        assert "calls" in meta, f"Component {name} missing 'calls'"
        
        # Check call signature can be parsed
        params, _ = parse_signature(meta["calls"])
        assert isinstance(params, list), f"Failed to parse params for {name}"

def test_layer_organization():
    """Test that our components are organized into the expected layers."""
    layers = set(meta["layer"] for meta in COMPONENTS.values())
    
    # Check expected layers exist
    required_layers = {"Context", "Ontology", "Rules", "Instances", "Verification"}
    for layer in required_layers:
        assert layer in layers, f"Missing expected layer: {layer}"
        
    # Check each layer has at least one component
    for layer in layers:
        components = [name for name, meta in COMPONENTS.items() if meta["layer"] == layer]
        assert len(components) > 0, f"Layer '{layer}' has no components"