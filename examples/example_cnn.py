"""
Example demonstrating the GenerateCNN and CNN capabilities of adAI
"""

import numpy as np
from adAI import GenerateCNN, CNN

np.random.seed(42)

print("--- 1. Data Generation ---")
n_samples = 120
height, width, channels = 14, 14, 1
X = np.random.randn(n_samples, height, width, channels).astype(np.float32)

pixel_sums = X.reshape(n_samples, -1).sum(axis=1)
y = np.zeros(n_samples, dtype=np.int32)
y[pixel_sums < -10] = 0
y[(pixel_sums >= -10) & (pixel_sums <= 10)] = 1
y[pixel_sums > 10] = 2

split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Generated data:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
print(f"  Class counts: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}, Class 2: {np.sum(y == 2)}")

print("\n--- 2. CNN Creation & Configuration ---")
cnn = GenerateCNN(
    input_shape=(height, width, channels),
    conv_layers=[(16, 3), (32, 3)],
    dense_layers=[32],
    output_size=3,
    learning_rate=0.02,
    epochs=50,
    l2=0.001,
    optimizer="radamw"
)

cnn.summary()

print("\n--- 3. CNN Training ---")
history = cnn.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    batch_size=16,
    verbose=True
)
cnn.print_training_summary(history)

print("\n--- 4. Model Evaluation ---")
metrics = cnn.evaluate(X_test, y_test)
cnn.print_metrics(metrics)

print("\n--- 5. Model Saving & Loading ---")
save_path = "saved_cnn.npz"
cnn.save(save_path)
print(f"CNN model saved successfully to '{save_path}'")

loaded_cnn = CNN.load(save_path)
print(f"CNN model loaded successfully from '{save_path}'")

sample_indices = [0, 1, 2]
test_samples = X_test[sample_indices]

preds_original = cnn.predict(test_samples)
preds_loaded = loaded_cnn.predict(test_samples)

print("\nPrediction verification on 3 sample inputs:")
for i, idx in enumerate(sample_indices):
    orig_pred = preds_original[i]
    load_pred = preds_loaded[i]
    true_label = y_test[idx]
    
    print(f"Sample {i+1} (True Class: {true_label}):")
    print(f"  Original predictions (probabilities): {orig_pred}")
    print(f"  Loaded predictions (probabilities):   {load_pred}")
    print(f"  Predicted class:                      {np.argmax(load_pred)}")
    assert np.allclose(orig_pred, load_pred, atol=1e-6)

print("\nAll checks passed successfully! Your NumPy CNN module is fully operational.")