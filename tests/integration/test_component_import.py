#!/usr/bin/env python
"""Test script for Cogitarelink DSPy integration."""

import sys
import importlib
from typing import List, Dict, Any

# Test direct import from Cogitarelink
print("Testing direct imports from Cogitarelink:")
try:
    import cogitarelink.core.context
    print("✅ Successfully imported cogitarelink.core.context")
    ctx = cogitarelink.core.context.ContextProcessor()
    print(f"  Created instance: {type(ctx).__name__}")
except Exception as e:
    print(f"❌ Failed to import cogitarelink.core.context: {e}")

try:
    import cogitarelink.verify.validator
    print("✅ Successfully imported cogitarelink.verify.validator")
    # Check if validate_entity exists
    if hasattr(cogitarelink.verify.validator, 'validate_entity'):
        print("  Found validate_entity function")
except Exception as e:
    print(f"❌ Failed to import cogitarelink.verify.validator: {e}")

# Test our component registry
print("\nTesting component registry:")
try:
    from cogitarelink_dspy.components import COMPONENTS
    print(f"✅ Successfully imported component registry with {len(COMPONENTS)} components")
    for name, meta in COMPONENTS.items():
        print(f"  - {name}: {meta['tool']} ({meta['layer']} layer)")
        # Verify module path exists
        module_path = meta['module']
        try:
            module_parts = module_path.split('.')
            if len(module_parts) > 1:
                parent_module = '.'.join(module_parts[:-1])
                try:
                    importlib.import_module(parent_module)
                    print(f"    ✅ Module {parent_module} exists")
                except ImportError as e:
                    print(f"    ❌ Module {parent_module} not found: {e}")
            else:
                try:
                    importlib.import_module(module_path)
                    print(f"    ✅ Module {module_path} exists")
                except ImportError as e:
                    print(f"    ❌ Module {module_path} not found: {e}")
        except Exception as e:
            print(f"    ❌ Error checking module {module_path}: {e}")
except Exception as e:
    print(f"❌ Failed to import component registry: {e}")

# Test individual wrapper creation (simpler than full get_tools)
print("\nTesting individual wrapper creation:")
try:
    from cogitarelink_dspy.wrappers import parse_signature
    context_component = COMPONENTS['ContextProcessor']
    call_sig = context_component['calls']
    
    print(f"Parsing signature: {call_sig}")
    try:
        params, return_type = parse_signature(call_sig)
        print(f"✅ Successfully parsed signature")
        print(f"  Parameters: {params}")
        print(f"  Return type: {return_type}")
    except Exception as e:
        print(f"❌ Error parsing signature: {e}")
        
except Exception as e:
    print(f"❌ Failed to test individual wrapper: {e}")

print("\nDone testing.")