"""
Test script to verify C++ backend integration with Python.
"""

import sys
import os

# Add the adAI source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'adAI'))

print("Testing C++ Backend Integration")
print("=" * 50)

# Test 1: Import the C++ module
print("\nTest 1: Importing adaicpp module...")
try:
    import adaicpp
    print("[PASS] adaicpp module imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import adaicpp: {e}")
    sys.exit(1)

# Test 2: Initialize the library
print("\nTest 2: Initializing library...")
try:
    adaicpp.initialize()
    print("[PASS] Library initialized")
except Exception as e:
    print(f"[FAIL] Failed to initialize: {e}")
    sys.exit(1)

# Test 3: Test device
print("\nTest 3: Testing device...")
try:
    device = adaicpp.cpu_device()
    print(f"[PASS] Got CPU device: {device.name()}")
except Exception as e:
    print(f"[FAIL] Failed to get device: {e}")
    sys.exit(1)

# Test 4: Test tensor creation
print("\nTest 4: Testing tensor creation...")
try:
    device = adaicpp.cpu_device()
    t = adaicpp.Tensor([2, 3], device)
    print(f"[PASS] Created tensor with shape: {t.shape}")
except Exception as e:
    print(f"[FAIL] Failed to create tensor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test tensor operations
print("\nTest 5: Testing tensor operations...")
try:
    device = adaicpp.cpu_device()
    a = adaicpp.Tensor([2, 2], device)
    a.fill(1.0)
    b = adaicpp.Tensor([2, 2], device)
    b.fill(2.0)
    c = adaicpp.add(a, b)
    print(f"[PASS] Addition successful")
except Exception as e:
    print(f"[FAIL] Failed tensor operations: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test graph creation
print("\nTest 6: Testing graph creation...")
try:
    graph = adaicpp.Graph()
    print("[PASS] Graph created")
except Exception as e:
    print(f"[FAIL] Failed to create graph: {e}")
    sys.exit(1)

# Test 7: Test placeholder
print("\nTest 7: Testing placeholder...")
try:
    x = graph.add_placeholder("x")
    print(f"[PASS] Placeholder added: {x.name}")
except Exception as e:
    print(f"[FAIL] Failed to add placeholder: {e}")
    sys.exit(1)

# Test 8: Test parameter
print("\nTest 8: Testing parameter...")
try:
    device = adaicpp.cpu_device()
    w = adaicpp.Tensor((2, 3), device)
    w.fill(1.0)
    param = graph.add_parameter("w", w)
    print(f"[PASS] Parameter added: {param.name}")
except Exception as e:
    print(f"[FAIL] Failed to add parameter: {e}")
    sys.exit(1)

# Test 9: Test graph compilation
print("\nTest 9: Testing graph compilation...")
try:
    graph.compile()
    print("[PASS] Graph compiled")
except Exception as e:
    print(f"[FAIL] Failed to compile graph: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("All tests passed! [SUCCESS]")
print("C++ backend is working correctly with Python.")
