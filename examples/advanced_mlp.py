"""
Advanced MLP Example - Multi-class classification with learning rate scheduling
"""

import numpy as np
from adAI import GenerateMLP, train_test_split, normalize, standardize, StepLR

# Generate synthetic multi-class classification data
np.random.seed(42)
n_samples = 1000
n_features = 10
n_classes = 3

X = np.random.randn(n_samples, n_features).astype(np.float32)
# Create 3 classes based on feature sums
y = np.zeros(n_samples, dtype=np.int32)
y[X.sum(axis=1) < -2] = 0
y[(X.sum(axis=1) >= -2) & (X.sum(axis=1) < 2)] = 1
y[X.sum(axis=1) >= 2] = 2

# One-hot encode targets
y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
y_onehot[np.arange(n_samples), y] = 1.0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Standardize features
X_train_std = standardize(X_train, axis=0)
X_test_std = standardize(X_test, axis=0)

# Create MLP with learning rate scheduler
mlp = GenerateMLP(
    input_size=n_features,
    hidden_layers=[64, 32],
    output_size=n_classes,
    learning_rate=0.01,
    epochs=200,
    activation="relu",
    output_activation="softmax",
    l2=0.001,
    dropout=0.2,
)

mlp.summary()

print("\nTraining with learning rate scheduling...")
scheduler = StepLR(step_size=50, gamma=0.5)

for epoch in range(200):
    # Update learning rate
    current_lr = scheduler.get_lr(mlp.learning_rate, epoch)
    mlp.learning_rate = current_lr
    
    # Train one epoch
    history = mlp.train(
        X_train_std, y_train,
        batch_size=32,
        verbose=False
    )
    
    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch + 1}/200, LR: {current_lr:.6f}, Loss: {history['final_loss']:.6f}")

mlp.print_training_summary(history)

# Evaluate
metrics = mlp.evaluate(X_test_std, y_test)
mlp.print_metrics(metrics)

# Make predictions on new data
new_samples = np.random.randn(5, n_features).astype(np.float32)
new_std = standardize(new_samples, axis=0)
predictions = mlp.predict(new_std)

print("\nPredictions for new samples:")
for i, pred in enumerate(predictions):
    predicted_class = np.argmax(pred)
    confidence = pred[predicted_class]
    print(f"Sample {i+1}: Class {predicted_class} (confidence: {confidence:.2%})")
