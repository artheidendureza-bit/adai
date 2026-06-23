import numpy as np
from adAI import GenerateMLP

np.random.seed(42)

n_samples = 200

size      = np.random.uniform(600, 4000, n_samples)
bedrooms  = np.random.randint(1, 6, n_samples).astype(float)
age       = np.random.uniform(0, 50, n_samples)
loc_score = np.random.uniform(1, 10, n_samples)

price = (
    0.12  * size
    + 20  * bedrooms    
    - 1.5 * age
    + 30  * loc_score
    + np.random.normal(0, 20, n_samples)
)
price = np.clip(price, 50, None)

X = np.column_stack([size, bedrooms, age, loc_score])
y = price

split = int(0.8 * n_samples)
indices = np.random.permutation(n_samples)
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)
X_test_norm  = (X_test  - X_min) / (X_max - X_min + 1e-8)

y_scale = y_train.max()
y_train_scaled = y_train / y_scale
y_test_scaled  = y_test  / y_scale

mlp = GenerateMLP(
    input_size=4,
    hidden_layers=[32, 16],
    output_size=1,
    learning_rate=0.05,
    epochs=2000,
    activation="relu",
    output_activation="linear",
)

mlp.summary()

print("\nTraining on housing data...")
history = mlp.train(
    X_train_norm, y_train_scaled,
    X_val=X_test_norm,
    y_val=y_test_scaled,
    batch_size=16,
    verbose=True,
)

mlp.print_training_summary(history)

print("\nEvaluation (scaled):")
metrics = mlp.evaluate(X_test_norm, y_test_scaled)
mlp.print_metrics(metrics)

new_houses = np.array([
    [2000, 3, 10, 8],
    [1200, 2, 30, 5],
    [3500, 5,  2, 9],
])

new_norm = (new_houses - X_min) / (X_max - X_min + 1e-8)
preds_scaled = mlp.predict(new_norm)

preds = preds_scaled * y_scale

print("\nPredictions for new houses:")
print(f"  {'Size':>6}  {'Beds':>4}  {'Age':>4}  {'Loc':>4}  |  {'Predicted Price':>16}")
print(f"  {'---':>6}  {'---':>4}  {'---':>4}  {'---':>4}  |  {'---':>16}")
for house, pred in zip(new_houses, preds):
    print(f"  {house[0]:6.0f}  {house[1]:4.0f}  {house[2]:4.0f}  {house[3]:4.0f}  |  ${pred * 1000:>14,.2f}")

print("\nSample test-set comparisons:")
test_preds_scaled = mlp.predict(X_test_norm)
test_preds = test_preds_scaled * y_scale

print(f"  {'Actual':>12}  {'Predicted':>12}  {'Error':>10}")
print(f"  {'---':>12}  {'---':>12}  {'---':>10}")
for actual, predicted in zip(y_test[:10], test_preds[:10]):
    err = predicted - actual
    print(f"  ${actual * 1000:>10,.0f}  ${predicted * 1000:>10,.0f}  ${err * 1000:>+9,.0f}")