from adAI import GenerateCNN, load_synthetic_cnn_data, load_data

X, y = load_synthetic_cnn_data(n_samples=120, height=14, width=14, channels=1, classes=3, random_state=42)
X_train, X_test, y_train, y_test, _ = load_data(X=X, y=y, random_state=42)

cnn = GenerateCNN(input_shape=(14, 14, 1), conv_layers=[(16, 3), (32, 3)], dense_layers=[32], output_size=3, epochs=50)
cnn.train(X_train, y_train, X_val=X_test, y_val=y_test, verbose=True)

metrics = cnn.evaluate(X_test, y_test)
print(f"Test Accuracy: {metrics['accuracy']:.2%}")