"""
Test script for adAI C++ backend integration
"""

import numpy as np
import sys

def test_cpp_backend_import():
    """Test if C++ backend can be imported"""
    print("Testing C++ backend import...")
    try:
        import adAI
        print(f"✓ adAI imported successfully")
        print(f"✓ C++ backend available: {adAI.is_cpp_backend_available()}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import adAI: {e}")
        return False

def test_cpp_graph_creation():
    """Test creating a C++ graph"""
    print("\nTesting C++ graph creation...")
    try:
        import adAI
        
        if not adAI.is_cpp_backend_available():
            print("⊘ C++ backend not available, skipping")
            return True
        
        graph = adAI.create_graph()
        print(f"✓ Created C++ graph: {graph}")
        return True
    except Exception as e:
        print(f"✗ Failed to create C++ graph: {e}")
        return False

def test_cpp_graph_operations():
    """Test basic graph operations"""
    print("\nTesting C++ graph operations...")
    try:
        import adAI
        
        if not adAI.is_cpp_backend_available():
            print("⊘ C++ backend not available, skipping")
            return True
        
        graph = adAI.create_graph()
        
        # Add placeholders
        x = graph.add_placeholder("x")
        y = graph.add_placeholder("y")
        print(f"✓ Added placeholders: x, y")
        
        # Add parameters
        w = graph.add_parameter("w", np.ones((2, 3), dtype=np.float32))
        print(f"✓ Added parameter: w")
        
        # Add operations
        add_op = graph.add_op("add", "add", [x, y])
        print(f"✓ Added operation: add")
        
        # Compile
        graph.compile()
        print(f"✓ Compiled graph")
        
        # Forward pass
        feeds = {
            "x": np.ones((2, 3), dtype=np.float32),
            "y": np.ones((2, 3), dtype=np.float32) * 2.0
        }
        results = graph.forward(feeds)
        print(f"✓ Forward pass completed")
        print(f"  Result shape: {results.get('add').shape if 'add' in results else 'N/A'}")
        
        return True
    except Exception as e:
        print(f"✗ Failed graph operations: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cpp_tensor_operations():
    """Test tensor operations"""
    print("\nTesting C++ tensor operations...")
    try:
        import adAI
        
        if not adAI.is_cpp_backend_available():
            print("⊘ C++ backend not available, skipping")
            return True
        
        ops = adAI.tensor_operations()
        print(f"✓ Available operations: {list(ops.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Failed tensor operations: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("adAI C++ Backend Integration Test")
    print("=" * 60)
    
    tests = [
        test_cpp_backend_import,
        test_cpp_graph_creation,
        test_cpp_graph_operations,
        test_cpp_tensor_operations,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
