"""
Complete Perceptron Example - Minimal & Clean
"""

import numpy as np
from adAI import GeneratePerceptron

perceptron = GeneratePerceptron(input_size=5, learning_rate=0.01, epochs=5000)
perceptron.summary()

data = perceptron.generate_data(n_train=200, n_val=50, n_test=50)

print("\nTraining...")
history = perceptron.train(data['X_train'], data['y_train'], 
X_val=data['X_val'], y_val=data['y_val'],
batch_size=128, early_stopping=True, patience=10, verbose=True)
perceptron.print_training_summary(history)

metrics = perceptron.evaluate(data['X_test'], data['y_test'])
perceptron.print_metrics(metrics)

model_path = "saved_perceptron.npz"
perceptron.save(model_path)
print(f"\nModel saved to {model_path}")

loaded = type(perceptron).load(model_path)
loaded_metrics = loaded.evaluate(data['X_test'], data['y_test'])
print(f"\nLoaded model test accuracy: {loaded_metrics['accuracy']:.2%}")

sample = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
pred = loaded.predict(sample)[0]
print(f"\nPrediction for {sample[0]}: {pred:.4f} ({'Positive' if pred > 0.5 else 'Negative'})")