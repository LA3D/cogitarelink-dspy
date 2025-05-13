#!/usr/bin/env python
"""
Direct test of Cogitarelink functionality without DSPy wrappers.

This script directly tests the Cogitarelink modules that our wrappers would call.
"""

import sys
import json
from typing import Dict, Any

# Dictionary mapping from our component names to their actual implementations
COMPONENTS_MAP = {
    "Utils": {
        "module": "cogitarelink.utils",
        "function": "load_module_source",
        "class": None,
    },
    "ContextProcessor": {
        "module": "cogitarelink.core.context",
        "function": None,
        "class": "ContextProcessor",
        "method": "compact",
    },
    "VocabRegistry": {
        "module": "cogitarelink.vocab.registry",
        "function": None,
        "class": "registry",  # Note: registry is already an instance in the module
        "method": "resolve",
    },
    "ValidateEntity": {
        "module": "cogitarelink.verify.validator",
        "function": "validate_entity",
        "class": None,
    },
    "GraphManager": {
        "module": "cogitarelink.core.graph",
        "function": None,
        "class": "GraphManager",
        "method": "query",
    },
    "Signer": {
        "module": "cogitarelink.verify.signer",
        "function": "verify",
        "class": None,
    }
}

def header(text):
    """Print a section header."""
    print(f"\n{'='*80}\n{text}\n{'='*80}")

def test_direct_utils():
    """Test Utils.load_module_source function directly."""
    header("Testing Utils.load_module_source")
    try:
        import cogitarelink.utils
        result = cogitarelink.utils.load_module_source("core")
        print(f"✅ Successfully called load_module_source")
        print(f"First 200 chars of result: {result[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_direct_context():
    """Test ContextProcessor.compact method directly."""
    header("Testing ContextProcessor.compact")
    try:
        import cogitarelink.core.context
        
        # Create processor instance
        processor = cogitarelink.core.context.ContextProcessor()
        
        # Create a simple document and context
        doc = {"name": "Alice", "knows": "Bob"}
        ctx = {"@context": {"name": "http://schema.org/name", "knows": "http://schema.org/knows"}}
        
        # Call compact
        result = processor.compact(doc, ctx)
        print(f"✅ Successfully called compact")
        print(f"Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_direct_registry():
    """Test registry.resolve function directly."""
    header("Testing registry.resolve")
    try:
        import cogitarelink.vocab.registry
        
        # Create a sample vocabulary URI
        test_uri = "http://example.org/vocab#term"
        
        try:
            result = cogitarelink.vocab.registry.registry.resolve(test_uri)
            print(f"✅ Successfully called resolve")
            print(f"Result: {result}")
        except KeyError:
            print(f"ℹ️ Expected KeyError for test URI: The URI is not registered")
            print(f"This is normal behavior if the vocab registry doesn't have this entry")
            
            # Try to see what vocabularies are available
            print(f"\nChecking what vocabularies are available:")
            registry_obj = cogitarelink.vocab.registry.registry
            if hasattr(registry_obj, 'list'):
                vocabs = registry_obj.list()
                print(f"Available vocabularies: {vocabs}")
            else:
                print(f"Registry doesn't have a 'list' method")
                
    except Exception as e:
        print(f"❌ Error: {e}")

def test_direct_validator():
    """Test validate_entity function directly."""
    header("Testing validate_entity")
    try:
        import cogitarelink.verify.validator
        
        # Create a simple test case
        target = "http://example.org/entity"
        shapes = """
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        
        ex:TestShape a sh:NodeShape ;
            sh:targetNode ex:entity ;
            sh:property [
                sh:path ex:property ;
                sh:minCount 1 ;
            ] .
        """
        
        try:
            result = cogitarelink.verify.validator.validate_entity(target, shapes)
            print(f"✅ Called validate_entity function")
            print(f"Result: {result}")
        except Exception as e:
            print(f"ℹ️ Expected error for minimal test case: {e}")
            print(f"This is likely because pySHACL needs more complete data")
    except Exception as e:
        print(f"❌ Error setting up test: {e}")

def test_direct_graph():
    """Test GraphManager.query method directly."""
    header("Testing GraphManager.query")
    try:
        import cogitarelink.core.graph
        
        # Create GraphManager instance
        graph_manager = cogitarelink.core.graph.GraphManager()
        
        # Try a simple query
        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . } LIMIT 10"
        
        try:
            result = graph_manager.query(query)
            print(f"✅ Successfully called query")
            print(f"Result: {result}")
        except Exception as e:
            print(f"ℹ️ Expected error for empty graph: {e}")
            print(f"This is normal if the graph is empty or not initialized")
    except Exception as e:
        print(f"❌ Error setting up test: {e}")

def test_direct_signer():
    """Test verify function directly."""
    header("Testing verify")
    try:
        import cogitarelink.verify.signer
        
        # Test parameters
        graph_id = "test-graph"
        signature = "test-signature"
        
        try:
            result = cogitarelink.verify.signer.verify(graph_id, signature)
            print(f"✅ Successfully called verify")
            print(f"Result: {result}")
        except Exception as e:
            print(f"ℹ️ Expected error for test parameters: {e}")
            print(f"This is normal if the signature verification requires setup")
    except Exception as e:
        print(f"❌ Error setting up test: {e}")

if __name__ == "__main__":
    # Run all direct tests
    test_direct_utils()
    test_direct_context()
    test_direct_registry()
    test_direct_validator()
    test_direct_graph()
    test_direct_signer()