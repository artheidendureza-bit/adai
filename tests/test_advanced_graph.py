import numpy as np
import pytest
import os
from adAI.graph import Graph, InteractiveExecutor, OpNode


def test_autograd():
    g = Graph()
    x = g.add_placeholder("x")
    w = g.add_parameter("w", np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
    b = g.add_parameter("b", np.array([1.0, 2.0], dtype=np.float32))
    
    xw = g.add_op("xw", "matmul", [x, w])
    y = g.add_op("y", "add", [xw, b])
    
    x_val = np.array([[1.0, 2.0]], dtype=np.float32)
    res = g.forward({"x": x_val})
    
    assert np.allclose(res["y"], np.array([[11.0, 15.0]]))
    
    g.backward("y")
    
    assert np.allclose(b.grad, np.array([[1.0, 1.0]]))
    assert np.allclose(w.grad, np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert np.allclose(x.grad, np.array([[5.0, 9.0]]))


def test_graph_optimization():
    g = Graph()
    x = g.add_placeholder("x")
    
    c1 = g.add_parameter("c1", np.array([3.0]))
    c2 = g.add_parameter("c2", np.array([4.0]))
    
    c3 = g.add_op("c3", "add", [c1, c2])
    y = g.add_op("y", "mul", [x, c3])
    
    dead = g.add_parameter("dead", np.array([99.0]))
    dead_op = g.add_op("dead_op", "add", [dead, c1])
    
    g.optimize(outputs=["y"])
    
    assert "dead" not in g.nodes
    assert "dead_op" not in g.nodes
    
    assert c3.name in g.nodes
    assert g.get_node("c3").__class__.__name__ == "ParameterNode"
    assert np.allclose(g.get_node("c3").value, np.array([7.0]))
    
    res = g.forward({"x": np.array([2.0])})
    assert np.allclose(res["y"], np.array([14.0]))


def test_interactive_execution_and_breakpoints():
    g = Graph()
    x = g.add_placeholder("x")
    w = g.add_parameter("w", np.array([2.0]))
    
    y1 = g.add_op("y1", "mul", [x, w])
    y2 = g.add_op("y2", "add", [y1, w])
    
    feeds = {"x": np.array([5.0])}
    executor = InteractiveExecutor(g, feeds)
    
    executor.add_breakpoint("y2")
    
    insights = executor.run_to_breakpoint()
    assert len(insights) == 1
    assert insights[0]["node_name"] == "y1"
    assert np.allclose(g.get_node("y1").value, np.array([10.0]))
    assert g.get_node("y2").value is None
    
    assert executor.has_next() is True
    step2 = executor.step()
    assert step2["node_name"] == "y2"
    assert np.allclose(g.get_node("y2").value, np.array([12.0]))
    assert executor.has_next() is False


def test_serialization(tmp_path):
    g = Graph()
    x = g.add_placeholder("x")
    w = g.add_parameter("w", np.array([[2.0], [3.0]], dtype=np.float32))
    y = g.add_op("y", "matmul", [x, w])
    
    res1 = g.forward({"x": np.array([[5.0, 10.0]], dtype=np.float32)})
    
    path = os.path.join(tmp_path, "graph.json")
    g.save(path)
    
    g2 = Graph.load(path)
    
    res2 = g2.forward({"x": np.array([[5.0, 10.0]], dtype=np.float32)})
    assert np.allclose(res1["y"], res2["y"])
    assert g2.get_node("w").__class__.__name__ == "ParameterNode"
    assert g2.get_node("y").__class__.__name__ == "OpNode"


def test_parallel_execution():
    g = Graph()
    x = g.add_placeholder("x")
    w1 = g.add_parameter("w1", np.array([2.0]))
    w2 = g.add_parameter("w2", np.array([3.0]))
    
    y1 = g.add_op("y1", "mul", [x, w1])
    y2 = g.add_op("y2", "mul", [x, w2])
    out = g.add_op("out", "add", [y1, y2])
    
    feeds = {"x": np.array([10.0])}
    
    res_seq = g.forward(feeds, parallel=False)
    
    for node in g.nodes.values():
        if isinstance(node, OpNode):
            node.value = None
            node.dirty = True
            
    res_par = g.forward(feeds, parallel=True)
    
    assert np.allclose(res_seq["out"], res_par["out"])
    assert np.allclose(res_par["out"], np.array([50.0]))


def test_visualization():
    g = Graph()
    x = g.add_placeholder("x", device="GPU")
    w = g.add_parameter("w", np.array([2.0]))
    y = g.add_op("y", "mul", [x, w])
    
    mermaid = g.to_mermaid()
    assert "graph TD" in mermaid
    assert "GPU" in mermaid
    assert "mul" in mermaid
    
    g.print_layout()
