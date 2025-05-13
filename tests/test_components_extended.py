import pytest
from cogitarelink_dspy.components import (
    COMPONENTS, get_tools_by_layer, list_layers, validate_component_registry
)

class TestComponentRegistry:
    """Test suite for the component registry functionality."""
    
    def test_registry_has_required_fields(self):
        """Test that all components have the required fields."""
        errors = validate_component_registry()
        assert len(errors) == 0, f"Found errors in component registry: {errors}"
    
    def test_registry_has_required_layers(self):
        """Test that all expected layers are present in the registry."""
        layers = list_layers()
        required_layers = ["Context", "Ontology", "Rules", "Instances", "Verification"]
        for layer in required_layers:
            assert layer in layers, f"Missing required layer: {layer}"
    
    def test_tools_by_layer(self):
        """Test that we can correctly filter tools by layer."""
        # Test all layers to ensure each has at least one tool
        for layer in list_layers():
            layer_tools = get_tools_by_layer(layer)
            assert len(layer_tools) > 0, f"No tools found for layer: {layer}"
            # Verify all tools in result belong to the specified layer
            assert all(meta["layer"] == layer for meta in layer_tools.values()), \
                f"Found tools with incorrect layer in result for {layer}"
    
    def test_tool_names_are_valid_identifiers(self):
        """Test that all tool names are valid Python identifiers."""
        for name, meta in COMPONENTS.items():
            assert meta["tool"].isidentifier(), \
                f"Tool name '{meta['tool']}' for component '{name}' is not a valid Python identifier"
    
    def test_module_paths_exist(self):
        """Test that all components have a module path."""
        for name, meta in COMPONENTS.items():
            assert "module" in meta, f"Component '{name}' is missing a module path"
            assert meta["module"], f"Component '{name}' has an empty module path"
    
    def test_signatures_have_return_types(self):
        """Test that all component signatures have return types."""
        for name, meta in COMPONENTS.items():
            assert " -> " in meta["calls"], \
                f"Component '{name}' signature '{meta['calls']}' does not specify a return type"
    
    def test_no_duplicate_tool_names(self):
        """Test that there are no duplicate tool names in the registry."""
        tool_names = {}
        for name, meta in COMPONENTS.items():
            tool_name = meta["tool"]
            assert tool_name not in tool_names, \
                f"Duplicate tool name '{tool_name}' found in components '{name}' and '{tool_names[tool_name]}'"
            tool_names[tool_name] = name

if __name__ == "__main__":
    pytest.main()