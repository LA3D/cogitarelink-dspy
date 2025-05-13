import pytest
import dspy
import re
from typing import List, Dict, Any, Tuple
from cogitarelink_dspy.wrappers import (
    parse_signature, make_tool_wrappers, get_tools, 
    get_tool_by_name, group_tools_by_layer
)
from cogitarelink_dspy.components import COMPONENTS

class TestSignatureParser:
    """Test suite for the signature parser functionality."""
    
    def test_basic_signature_parsing(self):
        """Test parsing basic function signatures."""
        test_cases = [
            ("forward(message:str)", [("message", "str")], None),
            ("validate(subject:str, predicate:str)", 
             [("subject", "str"), ("predicate", "str")], None),
        ]
        
        for sig_str, expected_params, expected_return in test_cases:
            params, return_type = parse_signature(sig_str)
            assert params == expected_params
            assert return_type == expected_return
    
    def test_return_type_parsing(self):
        """Test parsing signatures with return types."""
        test_cases = [
            ("load(source:str) -> dict", [("source", "str")], "dict"),
            ("verify(graph:str, sig:str) -> bool", 
             [("graph", "str"), ("sig", "str")], "bool"),
        ]
        
        for sig_str, expected_params, expected_return in test_cases:
            params, return_type = parse_signature(sig_str)
            assert params == expected_params
            assert return_type == expected_return
    
    def test_complex_type_parsing(self):
        """Test parsing signatures with complex types."""
        test_cases = [
            ("fetch(urls:List[str]) -> Dict[str, Any]",
             [("urls", "List[str]")], "Dict[str, Any]"),
            ("process(data:Dict[str, List[int]]) -> Tuple[int, str]",
             [("data", "Dict[str, List[int]]")], "Tuple[int, str]"),
        ]
        
        for sig_str, expected_params, expected_return in test_cases:
            params, return_type = parse_signature(sig_str)
            assert params == expected_params
            assert return_type == expected_return
    
    def test_handle_nested_brackets(self):
        """Test parsing signatures with nested brackets that might contain commas."""
        sig_str = "complex(a:int, b:List[Tuple[str, int]], c:Dict[str, List[Dict[str, Any]]]) -> bool"
        expected_params = [
            ("a", "int"), 
            ("b", "List[Tuple[str, int]]"), 
            ("c", "Dict[str, List[Dict[str, Any]]]")
        ]
        expected_return = "bool"
        
        params, return_type = parse_signature(sig_str)
        assert params == expected_params
        assert return_type == expected_return


class TestToolWrappers:
    """Test suite for the tool wrapper generation functionality."""
    
    def test_wrapper_generation(self):
        """Test that we generate the correct number of tool wrappers."""
        tools = make_tool_wrappers()
        assert len(tools) == len(COMPONENTS), \
            f"Expected {len(COMPONENTS)} tools, got {len(tools)}"
    
    def test_wrapper_properties(self):
        """Test that generated wrappers have the correct properties."""
        tools = make_tool_wrappers()
        
        for tool in tools:
            # Find the component that generated this tool
            component_name = None
            component_meta = None
            for name, meta in COMPONENTS.items():
                if meta["tool"] == tool.__name__:
                    component_name = name
                    component_meta = meta
                    break
            
            assert component_name is not None, f"Could not find component for tool {tool.__name__}"
            
            # Check properties
            assert isinstance(tool.signature, dspy.Signature), \
                f"Tool {tool.__name__} does not have a DSPy signature"
            
            assert component_meta["layer"] == tool.layer, \
                f"Tool {tool.__name__} has incorrect layer: expected {component_meta['layer']}, got {tool.layer}"
            
            assert component_meta["module"] == tool.module_path, \
                f"Tool {tool.__name__} has incorrect module path"
            
            assert component_meta["doc"] in tool.__doc__, \
                f"Tool {tool.__name__} is missing documentation"
    
    def test_get_tools(self):
        """Test the get_tools function for retrieving tool wrappers."""
        tools = get_tools()
        assert len(tools) == len(COMPONENTS), \
            f"Expected {len(COMPONENTS)} tools from get_tools(), got {len(tools)}"
    
    def test_get_tool_by_name(self):
        """Test getting a specific tool by name."""
        # Get a tool name from the registry
        tool_name = next(iter(COMPONENTS.values()))["tool"]
        tool = get_tool_by_name(tool_name)
        
        assert tool is not None, f"Could not find tool with name {tool_name}"
        assert tool.__name__ == tool_name, \
            f"Expected tool with name {tool_name}, got {tool.__name__}"
        
        # Test with a non-existent tool name
        non_existent_tool = get_tool_by_name("NonExistentTool")
        assert non_existent_tool is None, \
            f"Expected None for non-existent tool, got {non_existent_tool}"
    
    def test_group_tools_by_layer(self):
        """Test grouping tools by their semantic layer."""
        grouped_tools = group_tools_by_layer()
        
        # Check that all layers from the component registry are present
        registry_layers = set(meta["layer"] for meta in COMPONENTS.values())
        grouped_layers = set(grouped_tools.keys())
        
        assert registry_layers == grouped_layers, \
            f"Layers mismatch: registry has {registry_layers}, grouped has {grouped_layers}"
        
        # Check that each tool is in the correct layer group
        for layer, tools in grouped_tools.items():
            for tool in tools:
                assert tool.layer == layer, \
                    f"Tool {tool.__name__} is in wrong layer group: expected {tool.layer}, got {layer}"
                
                # Verify that this tool actually comes from a component with this layer
                found = False
                for meta in COMPONENTS.values():
                    if meta["tool"] == tool.__name__ and meta["layer"] == layer:
                        found = True
                        break
                
                assert found, f"Tool {tool.__name__} in layer {layer} does not match any component"

if __name__ == "__main__":
    pytest.main()