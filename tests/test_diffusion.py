"""
Unit tests for the Diffusion Model module of adAI
"""

import pytest
import numpy as np
from adAI import GenerateDiffusionModel, DiffusionModel


def test_generate_diffusion_model_creation():
    """Test that GenerateDiffusionModel creates a DiffusionModel with expected parameters."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128, 256],
        n_timesteps=1000
    )
    
    assert diffusion is not None
    assert diffusion.data_dim == 64
    assert diffusion.hidden_dims == [128, 256]
    assert diffusion.n_timesteps == 1000
    assert diffusion.beta_start == 0.0001
    assert diffusion.beta_end == 0.02


def test_diffusion_model_invalid_data_dim():
    """Test that DiffusionModel raises error for invalid data_dim."""
    with pytest.raises(ValueError):
        GenerateDiffusionModel(data_dim=0, hidden_dims=[128])


def test_diffusion_model_invalid_n_timesteps():
    """Test that DiffusionModel raises error for invalid n_timesteps."""
    with pytest.raises(ValueError):
        GenerateDiffusionModel(data_dim=64, hidden_dims=[128], n_timesteps=0)


def test_diffusion_model_noise_schedule():
    """Test that noise schedule is created correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100
    )
    
    assert len(diffusion.betas) == 100
    assert len(diffusion.alphas) == 100
    assert len(diffusion.alpha_cumprod) == 100
    assert diffusion.betas[0] >= diffusion.beta_start
    assert diffusion.betas[-1] <= diffusion.beta_end


def test_diffusion_model_timestep_embedding():
    """Test that timestep embedding works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100
    )
    
    t = np.array([0, 50, 99])
    embedding = diffusion._get_timestep_embedding(t, 64)
    
    assert embedding.shape == (3, 64)


def test_diffusion_model_q_sample():
    """Test that forward diffusion process (q_sample) works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100
    )
    
    x_start = np.random.randn(10, 64).astype(np.float32)
    t = np.array([50] * 10)
    
    x_noisy, noise = diffusion._q_sample(x_start, t)
    
    assert x_noisy.shape == (10, 64)
    assert noise.shape == (10, 64)


def test_diffusion_model_denoise_network():
    """Test that denoising network works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100
    )
    
    x_t = np.random.randn(10, 64).astype(np.float32)
    t = np.array([50] * 10)
    
    predicted_noise = diffusion._denoise_network(x_t, t)
    
    assert predicted_noise.shape == (10, 64)


def test_diffusion_model_p_sample():
    """Test that reverse diffusion process (p_sample) works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100
    )
    
    x_t = np.random.randn(10, 64).astype(np.float32)
    t = np.array([50] * 10)
    
    x_prev = diffusion._p_sample(x_t, t)
    
    assert x_prev.shape == (10, 64)


def test_diffusion_model_train():
    """Test that DiffusionModel training works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100,
        epochs=5
    )
    
    X_train = np.random.randn(50, 64).astype(np.float32)
    history = diffusion.train(X_train, batch_size=16, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 5
    assert history['epochs_trained'] == 5


def test_diffusion_model_train_with_validation():
    """Test that DiffusionModel training with validation works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100,
        epochs=5
    )
    
    X_train = np.random.randn(50, 64).astype(np.float32)
    X_val = np.random.randn(10, 64).astype(np.float32)
    
    history = diffusion.train(X_train, X_val=X_val, batch_size=16, verbose=False)
    
    assert 'val_loss_history' in history
    assert len(history['val_loss_history']) == 5


def test_diffusion_model_sample():
    """Test that DiffusionModel sampling works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100
    )
    
    samples = diffusion.sample(n_samples=10)
    
    assert samples.shape == (10, 64)


def test_diffusion_model_sample_custom_shape():
    """Test that DiffusionModel sampling with custom shape works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100
    )
    
    samples = diffusion.sample(n_samples=10, shape=(10, 64))
    
    assert samples.shape == (10, 64)


def test_diffusion_model_predict():
    """Test that DiffusionModel prediction works (for compatibility)."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100
    )
    
    X = np.random.randn(10, 64).astype(np.float32)
    output = diffusion.predict(X)
    
    assert output.shape == (10, 64)


def test_diffusion_model_evaluate():
    """Test that DiffusionModel evaluation works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100
    )
    
    X = np.random.randn(10, 64).astype(np.float32)
    metrics = diffusion.evaluate(X)
    
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert metrics['mse'] >= 0


def test_diffusion_model_save_load():
    """Test that DiffusionModel save and load works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100,
        epochs=5
    )
    
    X_train = np.random.randn(50, 64).astype(np.float32)
    diffusion.train(X_train, batch_size=16, verbose=False)
    
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
        temp_path = f.name
    
    try:
        diffusion.save(temp_path)
        loaded_diffusion = DiffusionModel.load(temp_path)
        
        assert loaded_diffusion.data_dim == diffusion.data_dim
        assert loaded_diffusion.hidden_dims == diffusion.hidden_dims
        assert loaded_diffusion.n_timesteps == diffusion.n_timesteps
        assert len(loaded_diffusion.loss_history) == len(diffusion.loss_history)
    finally:
        import time
        time.sleep(0.1)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def test_diffusion_model_summary():
    """Test that DiffusionModel summary prints without error."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100
    )
    
    # Should not raise any exception
    diffusion.summary()


def test_diffusion_model_custom_beta_schedule():
    """Test that custom beta schedule works correctly."""
    diffusion = GenerateDiffusionModel(
        data_dim=64,
        hidden_dims=[128],
        n_timesteps=100,
        beta_start=0.0005,
        beta_end=0.05
    )
    
    assert diffusion.beta_start == 0.0005
    assert diffusion.beta_end == 0.05
    assert diffusion.betas[0] >= 0.0005
    assert diffusion.betas[-1] <= 0.05
