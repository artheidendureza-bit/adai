import numpy as np
from adAI import GenerateDiffusionModel

X = np.random.randn(100, 16).astype(np.float32)

diffusion = GenerateDiffusionModel(data_dim=16, hidden_dims=[32, 16], n_timesteps=100, epochs=50)
diffusion.train(X, batch_size=16, verbose=True)

samples = diffusion.sample(n_samples=10)
print(f"Generated samples shape: {samples.shape}")
