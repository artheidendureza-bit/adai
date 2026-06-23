from adAI import GenerateMLP, load_data

# Load built-in housing dataset
X_train, X_test, y_train, y_test, scaler = load_data('boston_housing', random_state=42)

# Train MLP
mlp = GenerateMLP(input_size=13, hidden_layers=[32, 16], output_size=1, epochs=2000)
mlp.train(X_train, y_train, X_val=X_test, y_val=y_test, verbose=True)

# Evaluate
metrics = mlp.evaluate(X_test, y_test)
print(f"\nTest MSE: {metrics['mse']:.6f}")
