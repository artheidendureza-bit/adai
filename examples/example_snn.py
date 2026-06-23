from adAI import GenerateSNN, load_synthetic_snn_data, load_data

X, y = load_synthetic_snn_data(n_samples=200, input_size=10, classes=2, random_state=42)
X_train, X_test, y_train, y_test, _ = load_data(X=X, y=y, random_state=42)

snn = GenerateSNN(input_size=10, hidden_layers=[8], output_size=1, time_steps=20, epochs=50)
snn.train(X_train, y_train, X_val=X_test, y_val=y_test, verbose=True)

metrics = snn.evaluate(X_test, y_test)
print(f"Test Accuracy: {metrics['accuracy']:.2%}")
