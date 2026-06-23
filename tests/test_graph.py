import numpy as np
import pytest
from adAI.graph import Graph, PlaceholderNode, ParameterNode, OpNode


def test_topological_sorting():
    """Verify that graph sorts nodes topologically and detects cycles."""
    g = Graph()
    x = g.add_placeholder("x")
    w = g.add_parameter("w", np.array([[2.0]]))
    
    y = g.add_op("y", lambda a, b: np.dot(a, b), [x, w])
    
    g.compile()
    
    assert g._sorted_nodes.index(x) < g._sorted_nodes.index(y)
    assert g._sorted_nodes.index(w) < g._sorted_nodes.index(y)

    g2 = Graph()
    a = PlaceholderNode("a")
    # Make a cycle artificially by adding cross-inputs
    b = OpNode("b", lambda x: x, [a])
    a.inputs.append(b)  # Cycle: a -> b -> a
    g2.add_node(a)
    g2.add_node(b)
    with pytest.raises(ValueError, match="Cycle detected"):
        g2.compile()


def test_static_caching_and_dynamic_evaluation():
    """Verify that static subgraphs are cached and not re-evaluated unless inputs change."""
    g = Graph()
    
    x = g.add_placeholder("x")
    
    w = g.add_parameter("w", np.array([[3.0]]))
    b = g.add_parameter("b", np.array([1.0]))
    
    y_static = g.add_op("y_static", lambda w_val, b_val: w_val + b_val, [w, b], is_static=True)
    
    y = g.add_op("y", lambda x_val, s_val: np.dot(x_val, s_val), [x, y_static])

    feeds1 = {"x": np.array([[2.0]])}
    res1 = g.forward(feeds1)
    
    assert np.allclose(res1["y_static"], np.array([[4.0]]))
    assert np.allclose(res1["y"], np.array([[8.0]]))
    assert y_static.eval_count == 1
    assert y.eval_count == 1

    res2 = g.forward(feeds1)
    assert np.allclose(res2["y"], np.array([[8.0]]))
    assert y_static.eval_count == 1
    assert y.eval_count == 2

    feeds2 = {"x": np.array([[5.0]])}
    res3 = g.forward(feeds2)
    assert np.allclose(res3["y"], np.array([[20.0]]))
    assert y_static.eval_count == 1
    assert y.eval_count == 3

    w.set_value(np.array([[4.0]]))
    res4 = g.forward(feeds2)
    assert np.allclose(res4["y_static"], np.array([[5.0]]))
    assert np.allclose(res4["y"], np.array([[25.0]]))
    assert y_static.eval_count == 2
    assert y.eval_count == 4


def test_dynamic_modifications():
    """Verify we can change node properties dynamically and graph responds correctly."""
    g = Graph()
    x = g.add_placeholder("x")
    w = g.add_parameter("w", np.array([2.0]))
    
    y = g.add_op("y", lambda a: a * 2, [w], is_static=True)
    
    g.forward({"x": np.array([10.0])})
    assert y.eval_count == 1

    g.forward({"x": np.array([10.0])})
    assert y.eval_count == 1

    g.modify_node_static("y", is_static=False)
    
    g.forward({"x": np.array([10.0])})
    assert y.eval_count == 2
    g.forward({"x": np.array([10.0])})
    assert y.eval_count == 3


def test_full_graph_compilation_and_dynamic_modification():
    """Verify that compiling the graph works and dynamic structural edits rebuild properly."""
    g = Graph()
    x = g.add_placeholder("x")
    w = g.add_parameter("w", np.array([3.0]))
    
    # y = x * w
    y = g.add_op("y", lambda a, b: a * b, [x, w])
    
    # Compile and execute
    res = g.forward({"x": np.array([4.0])})
    assert np.allclose(res["y"], np.array([12.0]))
    assert g._compiled is True
    
    # 1. Modify Inputs dynamically
    # Create new node z
    z = g.add_parameter("z", np.array([5.0]))
    # Change y inputs to [x, z] instead of [x, w]
    g.set_inputs("y", [x, z])
    
    # Graph should be uncompiled now
    assert g._compiled is False
    
    # Forward pass should recompile and run with new inputs
    res2 = g.forward({"x": np.array([4.0])})
    assert np.allclose(res2["y"], np.array([20.0]))
    assert g._compiled is True

    # 2. Remove Node dynamically
    # Add a dummy node that uses y
    out = g.add_op("out", lambda a: a + 1, [y])
    res3 = g.forward({"x": np.array([4.0])})
    assert np.allclose(res3["out"], np.array([21.0]))
    
    # Remove the dummy node
    g.remove_node("out")
    assert g._compiled is False
    
    # Run again, 'out' should no longer be in results
    res4 = g.forward({"x": np.array([4.0])})
    assert "out" not in res4
    assert np.allclose(res4["y"], np.array([20.0]))
    assert g._compiled is True