from adAI import GenerateTransformer, load_synthetic_sequence_data

X, y = load_synthetic_sequence_data(n_samples=100, vocab_size=1000, max_seq_len=16, random_state=42)

transformer = GenerateTransformer(vocab_size=1000, d_model=64, n_heads=4, n_layers=2, max_seq_len=16, epochs=50)
transformer.train(X, y, batch_size=16, verbose=True)

predictions = transformer.predict(X)
print(f"Predictions shape: {predictions.shape}")
