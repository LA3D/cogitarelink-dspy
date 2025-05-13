# Component Catalog Implementation Plan

This document outlines the step-by-step process for implementing the Cogitarelink component catalog and DSPy tool wrappers in Jeremy Howard's nbdev style - with a focus on exploratory development, clear documentation, and test-driven design.

## Overview

We'll build a structured catalog of all Cogitarelink tools organized by semantic layer, enabling the agent to understand the 4-layer Semantic Web architecture:

1. **Context** - Working with JSON-LD contexts and namespaces
2. **Ontology** - Using vocabularies and ontology terms
3. **Rules** - Applying validation rules and shapes
4. **Instances** - Managing actual data instances
5. **Verification** - Verifying and signing graphs

Each component will include metadata about its layer, purpose, documentation, and call signature.

## Implementation Plan

### Step 1: Create component catalog notebook (01_components.ipynb)

Start with a new notebook that defines the component registry:

```markdown
# Component Catalog

> A centralized registry of all Cogitarelink tools organized by semantic layer.

This module provides a single dictionary that defines all available tools in the system, their documentation, and their layer in the semantic web stack. It serves as the single source of truth for tool discovery and documentation.
```

```python
#| default_exp COMPONENTS
#| hide
from nbdev.showdoc import *
```

```python
#| export

# Component registry organized by semantic layer:
# 1. Context - Tools for handling JSON-LD contexts and namespaces
# 2. Ontology - Tools for fetching and exploring ontologies/vocabularies
# 3. Rules - Tools for validation and rules processing
# 4. Instances - Tools for data instance management
# 5. Verification - Tools for verification and signatures

COMPONENTS = {
    # ===== Basic Echo Tool (for testing) =====
    "Echo": {
        "layer": "Utility",
        "tool": "EchoMessage",
        "doc": "Simply echoes the input message back.",
        "calls": "forward(message:str)"
    },
    
    # ===== Context Layer =====
    "SimpleContext": {
        "layer": "Context",
        "tool": "LoadContext",
        "doc": "Loads a simple JSON-LD context from a string or URL.",
        "calls": "load(source:str)"
    },
    
    # ===== Ontology Layer =====
    "OntologyFetcher": {
        "layer": "Ontology",
        "tool": "FetchOntology",
        "doc": "Fetches and caches a vocabulary or ontology by URI.",
        "calls": "fetch(uri:str)"
    },
    
    # ===== Rules Layer =====
    "SimpleValidator": {
        "layer": "Rules",
        "tool": "ValidateTriple",
        "doc": "Validates if a triple conforms to basic RDF rules.",
        "calls": "validate(subject:str, predicate:str, object:str)"
    },
    
    # ===== Instances Layer =====
    "TripleStore": {
        "layer": "Instances",
        "tool": "StoreTriple",
        "doc": "Stores a triple in the graph manager.",
        "calls": "add(subject:str, predicate:str, object:str)"
    },
    
    # ===== Verification Layer =====
    "SignatureChecker": {
        "layer": "Verification",
        "tool": "VerifySignature",
        "doc": "Verifies a digital signature on a named graph.",
        "calls": "verify(graph_id:str, signature:str)"
    }
}
```

```python
#| export
def get_tools_by_layer(layer, registry=COMPONENTS):
    """Return all tool definitions for a specific layer."""
    return {name: meta for name, meta in registry.items() 
            if meta['layer'] == layer}

def list_layers(registry=COMPONENTS):
    """Return all unique layers in the component registry."""
    return sorted(list(set(meta['layer'] for meta in registry.values())))
```

```python
# Test our component registry and helper functions
all_layers = list_layers()
print(f"Discovered layers: {all_layers}")
assert "Context" in all_layers
assert "Rules" in all_layers
assert "Instances" in all_layers

# Test getting tools by layer
utility_tools = get_tools_by_layer("Utility")
assert "Echo" in utility_tools
assert utility_tools["Echo"]["tool"] == "EchoMessage"

# Display a sample of documentation
for name, meta in list(COMPONENTS.items())[:2]:
    print(f"Tool: {meta['tool']} [Layer: {meta['layer']}]")
    print(f"Doc: {meta['doc']}")
    print(f"Calls: {meta['calls']}")
    print("-" * 40)
```

```python
#| export
def validate_component_registry(registry=COMPONENTS):
    """Validate that all entries in the component registry have required fields."""
    required_fields = ['layer', 'tool', 'doc', 'calls']
    errors = []
    
    for name, meta in registry.items():
        for field in required_fields:
            if field not in meta:
                errors.append(f"Component {name} is missing required field '{field}'")
                
    return errors

# Test the validation
errors = validate_component_registry()
assert len(errors) == 0, f"Found errors in component registry: {errors}"
```

```python
#| hide
import nbdev; nbdev.nbdev_export()
```

### Step 2: Create wrapper generator notebook (02_wrappers.ipynb)

```markdown
# Tool Wrappers

> Automatically generate DSPy modules from our component registry.

This module takes the component registry and generates fully-typed DSPy modules for each tool, with proper signatures and documentation.
```

```python
#| default_exp wrappers
#| hide
from nbdev.showdoc import *
```

```python
#| export
import importlib
import inspect
import dspy
from cogitarelink_dspy.COMPONENTS import COMPONENTS
```

```python
#| export
def parse_signature(sig_str):
    """Parse a signature string like 'foo(a:str, b:int) -> str' into parameter names,
    types, and return type."""
    # Simple parser for demonstration purposes
    params_str = sig_str.split('(')[1].split(')')[0]
    params = []
    
    if params_str:
        for param in params_str.split(','):
            name, type_hint = param.strip().split(':')
            params.append((name.strip(), type_hint.strip()))
            
    return params

# Test the parser
params = parse_signature("forward(message:str)")
assert params == [('message', 'str')]

params = parse_signature("validate(subject:str, predicate:str, object:str)")
assert len(params) == 3
```

```python
#| export
def make_tool_wrappers(registry=COMPONENTS):
    """Generate DSPy Module classes for each tool in the registry."""
    tools = []
    
    for name, meta in registry.items():
        params = parse_signature(meta['calls'])
        param_sig = " ".join(f"{p[0]}:{p[1]}" for p in params)
        signature = dspy.Signature(f"{param_sig} -> result")
        
        # Create a new DSPy Module class
        class_doc = f"{meta['doc']} [Layer: {meta['layer']}]"
        
        class ToolWrapper(dspy.Module):
            """Docstring will be replaced."""
            def forward(self, **kwargs):
                # This is just a stub - would actually call real implementation
                print(f"Called {meta['tool']} with args: {kwargs}")
                return f"Result from {meta['tool']}"
        
        ToolWrapper.__doc__ = class_doc
        ToolWrapper.__name__ = meta['tool']
        ToolWrapper.__qualname__ = meta['tool']
        ToolWrapper.signature = signature
        
        tools.append(ToolWrapper)
    
    return tools

# Generate the tools
TOOLS = make_tool_wrappers()
```

```python
# Test a sample tool
echo_tool = next(tool for tool in TOOLS if tool.__name__ == "EchoMessage")
print(f"Tool name: {echo_tool.__name__}")
print(f"Documentation: {echo_tool.__doc__}")

# Create an instance and test it
instance = echo_tool()
result = instance(message="Testing 123")
print(f"Result: {result}")
```

```python
#| hide
import nbdev; nbdev.nbdev_export()
```

### Step 3: Create a simple semantic agent notebook (03_semantic_agent.ipynb)

```markdown
# Hello-World Semantic-Web Agent

> A minimal agent implementing the 4-layer semantic web architecture.

This notebook brings together the component catalog and tool wrappers to create a simple agent that understands the layer-based architecture.
```

```python
#| default_exp semantic_agent
#| hide
from nbdev.showdoc import *
```

```python
#| export
import dspy
from cogitarelink_dspy.core import default_lm
from cogitarelink_dspy.wrappers import TOOLS

# Define the system prompt that explains the 4-layer architecture
SEM_WEB_SYSTEM = """
You are a Semantic-Web agent that reasons over a 4-layer architecture:
1. Context - Working with JSON-LD contexts and namespaces
2. Ontology - Using vocabularies and ontology terms
3. Rules - Applying validation rules and shapes
4. Instances - Managing actual data instances

Every tool is tagged with its PRIMARY layer. When answering a user question,
pick the tool from the HIGHEST layer that suffices to answer the question.
"""

# Configure the LLM with our system prompt
semantic_lm = dspy.LM("openai/gpt-4o-mini", system=SEM_WEB_SYSTEM)
```

```python
#| export
def make_semantic_agent(llm=None):
    """Create a Semantic Web agent that understands the 4-layer architecture."""
    lm = llm or semantic_lm
    
    # Create a StructuredAgent with all our tools
    agent = dspy.StructuredAgent(
        tools=TOOLS,
        lm=lm
    )
    
    return agent
```

```python
# Test the agent with a few simple queries
agent = make_semantic_agent()

queries = [
    "Echo back the phrase 'Hello Semantic Web'",
    "Load a context from 'https://schema.org/'",
    "Fetch the FOAF ontology",
    "Validate if 'ex:Alice ex:knows ex:Bob' is valid",
    "Store the triple 'ex:Earth ex:hasMoon ex:Moon'",
    "Verify the signature on graph 'g1'"
]

for query in queries:
    print(f"\nQuery: {query}")
    response = agent.query(query)
    print(f"Response: {response}\n")
    print(f"Tool used: {response.get('tool', 'unknown')}")
    print("-" * 60)
```

```python
#| hide
import nbdev; nbdev.nbdev_export()
```

### Step 4: Create tests for automatic verification

Create test files in the tests/ directory:

```python
# tests/test_components.py
import pytest
from cogitarelink_dspy.COMPONENTS import COMPONENTS, list_layers, get_tools_by_layer, validate_component_registry

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
```

```python
# tests/test_wrappers.py
import pytest
import dspy
from cogitarelink_dspy.wrappers import make_tool_wrappers, parse_signature, TOOLS
from cogitarelink_dspy.COMPONENTS import COMPONENTS

def test_signature_parser():
    """Test the signature parser correctly extracts parameters."""
    params = parse_signature("foo(a:str, b:int)")
    assert params == [("a", "str"), ("b", "int")]

def test_wrapper_generation():
    """Test that we generate the correct number of tool wrappers."""
    tools = make_tool_wrappers()
    assert len(tools) == len(COMPONENTS), "Number of tools doesn't match number of components"

def test_wrapper_properties():
    """Test that generated wrappers have correct properties."""
    echo_tool = next(tool for tool in TOOLS if tool.__name__ == "EchoMessage")
    assert isinstance(echo_tool.signature, dspy.Signature)
    assert "Utility" in echo_tool.__doc__
```

## Execution Timeline

1. **Day 1**: Set up notebook structure, implement component catalog
2. **Day 2**: Create wrapper generator, test basic functionality 
3. **Day 3**: Build semantic agent, run integration tests
4. **Day 4**: Add more sophisticated tools and test across layers

## Success Criteria

- All notebooks export successfully with nbdev_export
- Tests pass for both component registry and wrappers
- Agent can correctly select tools based on the semantic layer
- Agent understands the 4-layer architecture when answering questions

## Next Steps After Implementation

1. Connect to real Cogitarelink services when available
2. Expand the catalog with more domain-specific tools
3. Add few-shot examples for layer selection optimization
4. Build reflection memory for the agent to learn from past interactions

This implementation plan follows Jeremy Howard's approach of literate, exploratory programming with nbdev - making the development process transparent, testable, and well-documented.