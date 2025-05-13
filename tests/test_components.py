"""Tests for the components module."""

import pytest
from cogitarelink_dspy.components import (
    COMPONENTS, 
    list_layers, 
    get_tools_by_layer, 
    validate_component_registry
)

def test_registry_has_required_layers():
    """Test that all expected layers are present in the registry."""
    layers = list_layers()
    required_layers = ["Context", "Ontology", "Rules", "Instances", "Verification"]
    for layer in required_layers:
        assert layer in layers, f"Missing required layer: {layer}"

def test_tools_by_layer():
    """Test that we can correctly filter tools by layer."""
    context_tools = get_tools_by_layer("Context")
    assert len(context_tools) > 0, "No Context layer tools found"
    assert all(meta["layer"] == "Context" for meta in context_tools.values())

def test_registry_validation():
    """Test that the registry validation catches issues."""
    errors = validate_component_registry()
    assert len(errors) == 0, f"Registry validation found errors: {errors}"
    
    # Create a copy with a missing field to test validation
    test_registry = COMPONENTS.copy()
    test_registry["TestComponent"] = {
        "layer": "Context",
        # Missing "tool" field
        "doc": "A test component",
        "calls": "test()"
    }
    
    errors = validate_component_registry(test_registry)
    assert len(errors) == 1, "Validation should have found exactly one error"
    assert "missing required field 'tool'" in errors[0]