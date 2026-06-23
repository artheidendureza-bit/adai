"""
Computational graph representations for hybrid static/dynamic execution.
Caches static subgraphs and dynamically evaluates or rebuilds dynamic ones.
Supports autograd, graph optimization, interactive execution, serialization, parallel execution, and visualization.
"""

import numpy as np
import time
import json
from typing import List, Dict, Any, Callable, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor
from .graph_ops import OP_REGISTRY
from .backend import from_numpy, to_numpy, zeros, Backend, get_backend


class Node:
    """Base class for all nodes in the computational graph."""

    def __init__(
        self,
        name: str,
        inputs: Optional[List["Node"]] = None,
        is_static: bool = True,
        device: str = "CPU",
    ):
        self.name = name
        self.inputs = inputs or []
        self.is_static = is_static
        self.device = device
        self.value = None
        self.grad = None
        self.dirty: bool = True
        self.consumers: List["Node"] = []
        
        for input_node in self.inputs:
            input_node.consumers.append(self)

    def mark_dirty(self):
        """Mark this node and all of its consumers as dirty."""
        if not self.dirty:
            self.dirty = True
            for consumer in self.consumers:
                consumer.mark_dirty()

    def evaluate(self):
        """Evaluate the node's value. Should be overridden by subclasses."""
        raise NotImplementedError


class PlaceholderNode(Node):
    """A node that represents external input, always treated as dynamic."""

    def __init__(self, name: str, device: str = "CPU"):
        super().__init__(name=name, inputs=[], is_static=False, device=device)

    def set_value(self, value):
        """Set the value of the placeholder and invalidate dependent nodes."""
        self.value = from_numpy(value)
        self.mark_dirty()
        self.dirty = False

    def evaluate(self):
        if self.value is None:
            raise ValueError(f"Placeholder {self.name} has no value set.")
        return self.value


class ParameterNode(Node):
    """A node representing model weights or biases, static unless updated."""

    def __init__(self, name: str, value, device: str = "CPU"):
        super().__init__(name=name, inputs=[], is_static=True, device=device)
        self.value = from_numpy(value)
        self.dirty = False  # Parameters start clean

    def set_value(self, value):
        self.value = from_numpy(value)
        self.mark_dirty()
        self.dirty = False

    def evaluate(self):
        return self.value


class OpNode(Node):
    """A node representing an operation applied to input nodes."""

    def __init__(
        self,
        name: str,
        op: Union[Callable, str],
        inputs: List[Node],
        is_static: Optional[bool] = None,
        grad_fn: Optional[Callable] = None,
        device: str = "CPU",
    ):
        # Resolve op if it is a string ID
        self.op_name: Optional[str] = None
        if isinstance(op, str):
            self.op_name = op
            if op not in OP_REGISTRY:
                raise ValueError(f"Operation '{op}' is not registered in OP_REGISTRY.")
            op_obj = OP_REGISTRY[op]
            forward_fn = op_obj.forward_fn
            if grad_fn is None:
                grad_fn = op_obj.backward_fn
        else:
            forward_fn = op

        # By default, an operation is static if all its inputs are static
        if is_static is None:
            is_static = all(inp.is_static for inp in inputs)
            
        super().__init__(name=name, inputs=inputs, is_static=is_static, device=device)
        self.op = forward_fn
        self.grad_fn = grad_fn
        self.eval_count = 0  # To track how many times this node was computed

    def evaluate(self) -> np.ndarray:
        if self.is_static and not self.dirty and self.value is not None:
            return self.value

        input_vals = [inp.evaluate() for inp in self.inputs]
        self.value = self.op(*input_vals)
        self.eval_count += 1
        
        if self.is_static:
            self.dirty = False
            
        return self.value


class Graph:
    """Manages the computational graph, topological sorting, optimizations, and execution."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self._sorted_nodes: List[Node] = []
        self._needs_sorting = True
        self._compiled = False
        self._compiled_instructions: List[tuple] = []

    def add_node(self, node: Node) -> Node:
        """Add a node to the graph and invalidate compilation."""
        if node.name in self.nodes:
            raise ValueError(f"Node with name {node.name} already exists in graph.")
        self.nodes[node.name] = node
        self._needs_sorting = True
        self._compiled = False
        return node

    def remove_node(self, name: str):
        """Remove a node from the graph and disconnect consumers."""
        if name not in self.nodes:
            raise ValueError(f"Node {name} does not exist in graph.")
        node = self.nodes[name]
        
        # Disconnect from inputs
        for inp in node.inputs:
            if node in inp.consumers:
                inp.consumers.remove(node)
                
        # Disconnect consumers
        for consumer in list(node.consumers):
            if node in consumer.inputs:
                consumer.inputs.remove(node)
            consumer.mark_dirty()
            
        del self.nodes[name]
        self._needs_sorting = True
        self._compiled = False

    def set_inputs(self, name: str, new_inputs: List[Union[Node, str]]):
        """Dynamically modify the inputs of an existing node."""
        node = self.get_node(name)
        
        # Disconnect from old inputs
        for inp in node.inputs:
            if node in inp.consumers:
                inp.consumers.remove(node)
                
        # Resolve and connect new inputs
        resolved_inputs = []
        for inp in new_inputs:
            if isinstance(inp, str):
                resolved_inputs.append(self.get_node(inp))
            else:
                resolved_inputs.append(inp)
                
        node.inputs = resolved_inputs
        for inp in resolved_inputs:
            inp.consumers.append(node)
            
        node.mark_dirty()
        self._needs_sorting = True
        self._compiled = False

    def get_node(self, name: str) -> Node:
        """Get a node by name."""
        return self.nodes[name]

    def add_placeholder(self, name: str, device: str = "CPU") -> PlaceholderNode:
        node = PlaceholderNode(name, device=device)
        self.add_node(node)
        return node

    def add_parameter(self, name: str, value, device: str = "CPU") -> ParameterNode:
        node = ParameterNode(name, value, device=device)
        self.add_node(node)
        return node

    def add_op(
        self,
        name: str,
        op: Union[Callable, str],
        inputs: List[Union[Node, str]],
        is_static: Optional[bool] = None,
        grad_fn: Optional[Callable] = None,
        device: str = "CPU",
    ) -> OpNode:
        resolved_inputs = []
        for inp in inputs:
            if isinstance(inp, str):
                resolved_inputs.append(self.get_node(inp))
            else:
                resolved_inputs.append(inp)
        node = OpNode(name, op, resolved_inputs, is_static, grad_fn, device)
        self.add_node(node)
        return node

    def compile(self):
        """Topologically sort nodes and compile the graph into an optimized execution pipeline."""
        if self._compiled and not self._needs_sorting:
            return

        # 1. Topological Sort
        if self._needs_sorting:
            visited = set()
            temp_visited = set()
            order = []

            def visit(node: Node):
                if node.name in temp_visited:
                    raise ValueError("Cycle detected in computational graph!")
                if node.name not in visited:
                    temp_visited.add(node.name)
                    for input_node in node.inputs:
                        visit(input_node)
                    temp_visited.remove(node.name)
                    visited.add(node.name)
                    order.append(node)

            for node in self.nodes.values():
                if node.name not in visited:
                    visit(node)

            self._sorted_nodes = order
            self._needs_sorting = False

        # 2. Build compiled instructions
        self._compiled_instructions = []
        for node in self._sorted_nodes:
            if isinstance(node, OpNode):
                self._compiled_instructions.append(
                    (node, node.op, node.inputs, node.is_static)
                )

        self._compiled = True

    def forward(self, feeds: Dict[str, Any], parallel: bool = False) -> Dict[str, Any]:
        """
        Executes the compiled graph forward pass.
        Reuses cached static subgraphs where inputs/parameters haven't changed.
        Supports sequential or parallel thread-pool execution.
        """
        self.compile()

        # Update placeholder values
        for name, value in feeds.items():
            node = self.get_node(name)
            if not isinstance(node, PlaceholderNode):
                raise TypeError(f"Node {name} is not a PlaceholderNode.")
            node.set_value(value)

        if not parallel:
            # Sequential execution
            for node, op, inputs, is_static in self._compiled_instructions:
                if is_static and not node.dirty and node.value is not None:
                    continue
                
                input_vals = [inp.value for inp in inputs]
                node.value = op(*input_vals)
                node.eval_count += 1
                if is_static:
                    node.dirty = False
        else:
            # Parallel execution using ThreadPoolExecutor
            # Build list of pending ops
            pending_nodes = set(node for node, _, _, _ in self._compiled_instructions)
            
            # Nodes that do not need to run because they're static and not dirty
            cached_nodes = set()
            for node, op, inputs, is_static in self._compiled_instructions:
                if is_static and not node.dirty and node.value is not None:
                    cached_nodes.add(node)
                    pending_nodes.remove(node)
            
            # Resolve immediate in-degrees (number of dependency nodes currently running/pending)
            in_degree = {}
            for node in pending_nodes:
                # Count dependencies that are in pending_nodes
                deps = sum(1 for inp in node.inputs if inp in pending_nodes)
                in_degree[node.name] = deps

            completed_nodes = set(self.nodes.values()) - pending_nodes
            
            with ThreadPoolExecutor() as executor:
                def execute_node_task(node_tuple):
                    nd, op, inputs, is_static = node_tuple
                    input_vals = [inp.value for inp in inputs]
                    nd.value = op(*input_vals)
                    nd.eval_count += 1
                    if is_static:
                        nd.dirty = False
                    return nd

                while pending_nodes:
                    # Find nodes with in-degree == 0
                    runnable = [
                        item for item in self._compiled_instructions
                        if item[0] in pending_nodes and in_degree[item[0].name] == 0
                    ]
                    
                    if not runnable:
                        # Safety check for cycles (though compile checks this)
                        break
                        
                    futures = {executor.submit(execute_node_task, item): item[0] for item in runnable}
                    
                    for fut in futures:
                        completed_node = fut.result()
                        pending_nodes.remove(completed_node)
                        completed_nodes.add(completed_node)
                        # Decrement in-degrees of consumers
                        for consumer in completed_node.consumers:
                            if consumer.name in in_degree:
                                in_degree[consumer.name] -= 1

        # Build results dictionary
        return {name: node.value for name, node in self.nodes.items() if node.value is not None}

    def backward(self, target_node_name: str):
        """Computes reverse-mode automatic differentiation gradients for all nodes."""
        self.compile()
        target = self.get_node(target_node_name)
        if target.value is None:
            raise ValueError(f"Cannot run backward on {target_node_name} as it has no computed value.")

        # Reset all gradients
        for node in self.nodes.values():
            node.grad = None

        # Seed target gradient
        target_grad = ones(target.shape)
        target.grad = target_grad

        # Walk backward through topologically sorted nodes
        for node in reversed(self._sorted_nodes):
            if node.grad is None or not isinstance(node, OpNode):
                continue

            if node.grad_fn is None:
                # No gradient defined, skip or warning
                continue

            # Compute gradients for inputs
            input_vals = [inp.value for inp in node.inputs]
            input_grads = node.grad_fn(node.grad, input_vals)

            from .backend import add
            for inp, g_val in zip(node.inputs, input_grads):
                if inp.grad is None:
                    inp.grad = from_numpy(g_val)
                else:
                    inp.grad = add(inp.grad, from_numpy(g_val))

    def optimize(self, outputs: List[str]):
        """
        Performs graph optimization:
        1. Dead Code Elimination (DCE): Removes nodes not contributing to the specified outputs.
        2. Constant Folding: Folds subgraphs of fully static nodes.
        """
        # --- 1. Dead Code Elimination ---
        target_nodes = [self.get_node(name) for name in outputs]
        reachable = set()

        def dfs(node: Node):
            if node not in reachable:
                reachable.add(node)
                for inp in node.inputs:
                    dfs(inp)

        for t_node in target_nodes:
            dfs(t_node)

        # Remove unreachable nodes
        unreachable_names = set(self.nodes.keys()) - {node.name for node in reachable}
        for name in unreachable_names:
            self.remove_node(name)

        # Re-compile to get updated topological sort
        self._needs_sorting = True
        self.compile()

        # --- 2. Constant Folding ---
        # Traverse nodes topologically. If an OpNode contains only static inputs, we can fold it.
        folded_any = True
        while folded_any:
            folded_any = False
            self.compile()
            for node in list(self._sorted_nodes):
                if isinstance(node, OpNode) and node.is_static:
                    # Evaluate the static value immediately
                    folded_value = node.evaluate()
                    folded_value_np = to_numpy(folded_value)
                    folded_node = ParameterNode(name=node.name, value=folded_value_np, device=node.device)
                    
                    # Replace OpNode with ParameterNode
                    self.nodes[node.name] = folded_node
                    folded_node.consumers = list(node.consumers)
                    
                    # Update input consumers lists
                    for inp in node.inputs:
                        if node in inp.consumers:
                            inp.consumers.remove(node)
                            
                    # Update consumer inputs lists
                    for consumer in node.consumers:
                        consumer.inputs = [folded_node if x == node else x for x in consumer.inputs]
                    
                    folded_any = True
                    self._needs_sorting = True
                    self._compiled = False
                    break  # Break to re-sort and repeat check safely

        self.compile()

    def modify_node_static(self, name: str, is_static: bool):
        """Dynamically toggle the static/dynamic property of a node and invalidate compile."""
        node = self.get_node(name)
        if node.is_static != is_static:
            node.is_static = is_static
            node.mark_dirty()
            self._compiled = False

    # --- Serialization ---
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph schema and parameters into a JSON-compatible dictionary."""
        node_dicts = []
        for name, node in self.nodes.items():
            node_type = node.__class__.__name__
            node_dict = {
                "name": node.name,
                "type": node_type,
                "inputs": [inp.name for inp in node.inputs],
                "is_static": node.is_static,
                "device": node.device,
            }
            if isinstance(node, ParameterNode) or isinstance(node, PlaceholderNode):
                if node.value is not None:
                    node_dict["value"] = to_numpy(node.value).tolist()
            if isinstance(node, OpNode):
                node_dict["op_name"] = node.op_name
            node_dicts.append(node_dict)
        return {"nodes": node_dicts}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Graph":
        """Reconstruct a Graph instance from a serialized dictionary."""
        g = cls()
        
        # 1. Create all nodes without connecting inputs first
        node_configs = data["nodes"]
        for config in node_configs:
            name = config["name"]
            ntype = config["type"]
            device = config.get("device", "CPU")
            is_static = config.get("is_static", True)
            
            if ntype == "PlaceholderNode":
                val = config.get("value")
                node = PlaceholderNode(name, device=device)
                if val is not None:
                    node.set_value(np.array(val, dtype=np.float32))
            elif ntype == "ParameterNode":
                val = config.get("value")
                node = ParameterNode(name, np.array(val, dtype=np.float32), device=device)
            elif ntype == "OpNode":
                op_name = config.get("op_name")
                # OpNode requires forward op definition, we use OP_REGISTRY or placeholder identity
                op = op_name if op_name else lambda *args: args[0]
                node = OpNode(name, op, [], is_static=is_static, device=device)
            else:
                raise ValueError(f"Unknown node type {ntype}")
                
            g.nodes[name] = node

        # 2. Wire up the inputs/consumers
        for config in node_configs:
            name = config["name"]
            node = g.get_node(name)
            input_names = config["inputs"]
            resolved_inputs = [g.get_node(in_name) for in_name in input_names]
            
            node.inputs = resolved_inputs
            for inp in resolved_inputs:
                inp.consumers.append(node)
                
        g._needs_sorting = True
        g._compiled = False
        return g

    def save(self, path: str):
        """Save the graph schema to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str) -> "Graph":
        """Load a graph schema from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # --- Visualization ---
    def to_mermaid(self) -> str:
        """Generate a Mermaid flowchart diagram showing graph dependencies, device, and cache status."""
        self.compile()
        lines = ["graph TD"]
        for node in self._sorted_nodes:
            # Visual styling
            device_str = f" [{node.device}]"
            static_str = " (Static)" if node.is_static else " (Dynamic)"
            
            if isinstance(node, PlaceholderNode):
                style = "fill:#e1f5fe,stroke:#0288d1,stroke-width:2px"
                label = f"{node.name} [Placeholder{device_str}]"
            elif isinstance(node, ParameterNode):
                style = "fill:#efebe9,stroke:#5d4037,stroke-width:2px"
                label = f"{node.name} [Parameter{device_str}{static_str}]"
            elif isinstance(node, OpNode):
                style = "fill:#e8f5e9,stroke:#388e3c,stroke-width:2px"
                op_lbl = node.op_name if node.op_name else "CustomOp"
                label = f"{node.name} [Op: {op_lbl}{device_str}{static_str}]"
            else:
                style = ""
                label = f"{node.name}"

            lines.append(f'    {node.name}["{label}"]')
            lines.append(f"    style {node.name} {style}")

            for consumer in node.consumers:
                lines.append(f"    {node.name} --> {consumer.name}")

        return "\n".join(lines)

    def print_layout(self) -> None:
        """Print a structured text visual layout of the graph execution order."""
        self.compile()
        print("\n=== Graph Compilation Layout ===")
        for index, node in enumerate(self._sorted_nodes):
            inputs_str = ", ".join(inp.name for inp in node.inputs) if node.inputs else "None"
            node_type = node.__class__.__name__
            op_str = f" (op: {node.op_name})" if isinstance(node, OpNode) and node.op_name else ""
            print(f"[{index:02d}] Node: {node.name:<12} | Type: {node_type:<15}{op_str:<12} | Inputs: [{inputs_str}] | Device: {node.device}")
        print("================================")


class InteractiveExecutor:
    """Provides step-by-step interactive execution and debugging tools for Graph."""

    def __init__(self, graph: Graph, feeds: Dict[str, Any]):
        self.graph = graph
        self.graph.compile()
        
        # Seed placeholders
        for name, value in feeds.items():
            node = self.graph.get_node(name)
            if not isinstance(node, PlaceholderNode):
                raise TypeError(f"Node {name} is not a PlaceholderNode.")
            node.set_value(value)

        self.instructions = self.graph._compiled_instructions
        self.step_index = 0
        self.breakpoints: Set[str] = set()

    def has_next(self) -> bool:
        return self.step_index < len(self.instructions)

    def step(self) -> Dict[str, Any]:
        """Execute the next instruction and return runtime telemetry insights."""
        if not self.has_next():
            raise IndexError("No more instructions to execute.")

        node, op, inputs, is_static = self.instructions[self.step_index]
        self.step_index += 1

        start_time = time.perf_counter()
        
        # Check cache if static
        cached_hit = False
        if is_static and not node.dirty and node.value is not None:
            cached_hit = True
        else:
            input_vals = [inp.value for inp in inputs]
            node.value = op(*input_vals)
            node.eval_count += 1
            if is_static:
                node.dirty = False

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        val_summary = ""
        if node.value is not None:
            val_summary = f"shape={node.value.shape}, dtype={node.value.dtype}, mean={np.mean(node.value):.4f}"

        return {
            "node_name": node.name,
            "type": node.__class__.__name__,
            "cached_hit": cached_hit,
            "latency_ms": elapsed_ms,
            "device": node.device,
            "value_summary": val_summary,
        }

    def add_breakpoint(self, node_name: str):
        self.breakpoints.add(node_name)

    def remove_breakpoint(self, node_name: str):
        if node_name in self.breakpoints:
            self.breakpoints.remove(node_name)

    def run_to_breakpoint(self) -> List[Dict[str, Any]]:
        """Run execution steps sequentially until hitting a breakpoint or completion."""
        insights = []
        while self.has_next():
            next_node = self.instructions[self.step_index][0]
            if next_node.name in self.breakpoints:
                break
            insights.append(self.step())
        return insights
