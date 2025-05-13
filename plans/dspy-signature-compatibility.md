# DSPy Signature Compatibility Issues

## Overview

This document outlines the compatibility issues encountered when integrating the Cogitarelink component registry with DSPy's Signature API. The primary error occurs during tool wrapper generation, preventing the successful creation of DSPy Module classes that wrap Cogitarelink functionality.

## Current Issues

### 1. Signature Field Error

**Error Message:**
```
TypeError: Field `signature` in `StringSignature` must be declared with InputField or OutputField, but field `signature` has `field.json_schema_extra=None`
```

This error occurs when attempting to set the `signature` attribute on dynamically created tool wrapper classes. The DSPy API expects specific field declarations that our current approach doesn't satisfy.

### 2. Unknown Type Name Errors

**Error Message:**
```
ValueError: Unknown name: Entity
```

When parsing parameter types, DSPy's type system has trouble with custom types like `Entity` from Cogitarelink. It expects types it can resolve in its own type system.

### 3. Complex Type Handling

For complex types like `Union[Entity, str]` or `Dict[str, List[Any]]`, the current parsing approach oversimplifies these types, which may lead to incorrect type checking or documentation.

## Root Causes

1. **DSPy's Signature Implementation**: DSPy uses a specific approach for creating signatures that requires proper declaration of input and output fields, which our dynamic class generation doesn't fully comply with.

2. **Type Resolution**: DSPy attempts to resolve type names at runtime, but custom types from Cogitarelink aren't in its scope.

3. **Signature String Parsing**: Our current approach parses signature strings manually, which might not align with DSPy's expected format.

## Potential Solutions

### Approach 1: Use DSPy's InputField and OutputField Explicitly

Rather than setting the signature as an attribute, we could directly define input and output fields using DSPy's API:

```python
class ToolWrapper(dspy.Module):
    input_param = dspy.InputField(description="Input parameter")
    output_result = dspy.OutputField(description="Output result")
```

### Approach 2: Custom Type Registration

Create a type registration system that maps Cogitarelink types to DSPy-compatible types:

```python
TYPE_MAPPINGS = {
    "Entity": "Any",
    "Union[Entity, str]": "str",
    # ...more mappings
}
```

### Approach 3: Subclass DSPy's Signature

Create a custom Signature subclass that handles our specific needs while maintaining compatibility:

```python
class CogitarelinkSignature(dspy.Signature):
    def __init__(self, signature_str, instructions=None):
        # Custom implementation
        pass
```

### Approach 4: Simplified Wrapper without Signatures

As a fallback, create wrappers that don't use DSPy's Signature API directly but still provide the tool functionality:

```python
class SimplifiedToolWrapper(dspy.Module):
    def __init__(self, meta):
        self.meta = meta
        
    def forward(self, **kwargs):
        # Implementation without signature
        pass
```

## Implementation Plan

1. **Investigate DSPy Signature API**: Perform a deeper dive into how DSPy's Signature API works, particularly focusing on dynamic class generation.

2. **Create Prototype Implementations**: For each approach, create a simplified prototype to test viability.

3. **Integration Testing**: Test each approach with the full Cogitarelink integration pipeline.

4. **Documentation**: Update the component registry and wrapper documentation to reflect the chosen approach.

5. **Refactor Existing Code**: Apply the solution to the current codebase.

## Decision Criteria

The chosen approach should:
- Maintain compatibility with DSPy's agent framework
- Preserve type information from Cogitarelink where possible
- Be maintainable as both DSPy and Cogitarelink evolve
- Handle errors gracefully in production environments

## Next Steps

1. Set up a focused testing environment for signature API experiments
2. Test each approach with simplified examples
3. Implement the most promising approach in the full codebase
4. Update integration tests to verify the solution