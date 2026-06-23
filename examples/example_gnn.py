import numpy as np
from adAI import GenerateGNN, load_synthetic_graph_data
from adAI.gnn import Graph

node_features, adj_matrix, labels = load_synthetic_graph_data(n_nodes=50, n_features=16, n_classes=4, random_state=42)
graph = Graph(node_features, adj_matrix)

gnn = GenerateGNN(input_dim=16, hidden_dims=[32], output_dim=4, epochs=50)
mask = np.array([True] * 25 + [False] * 25)  # First half labeled
gnn.train(graph, labels, mask=mask, verbose=True)

predictions = gnn.predict(graph)
print(f"Predictions shape: {predictions.shape}")
