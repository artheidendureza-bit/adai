"""
Unit tests for the Transformer module of adAI
"""

import pytest
import numpy as np
from adAI import GenerateTransformer, Transformer


def test_multi_head_attention_creation():
    """Test that MultiHeadAttention is created with correct parameters."""
    from adAI.transformer import MultiHeadAttention
    
    attention = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.1)
    
    assert attention.d_model == 512
    assert attention.n_heads == 8
    assert attention.d_k == 64
    assert attention.dropout == 0.1


def test_multi_head_attention_invalid_d_model():
    """Test that MultiHeadAttention raises error for invalid d_model."""
    from adAI.transformer import MultiHeadAttention
    
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=0, n_heads=8)


def test_multi_head_attention_invalid_n_heads():
    """Test that MultiHeadAttention raises error for invalid n_heads."""
    from adAI.transformer import MultiHeadAttention
    
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=512, n_heads=0)


def test_multi_head_attention_invalid_division():
    """Test that MultiHeadAttention raises error when d_model not divisible by n_heads."""
    from adAI.transformer import MultiHeadAttention
    
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=512, n_heads=7)


def test_multi_head_attention_forward():
    """Test that MultiHeadAttention forward pass works correctly."""
    from adAI.transformer import MultiHeadAttention
    
    attention = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    x = np.random.randn(2, 10, 64).astype(np.float32)
    
    output = attention.forward(x)
    
    assert output.shape == (2, 10, 64)


def test_feed_forward_creation():
    """Test that FeedForward is created with correct parameters."""
    from adAI.transformer import FeedForward
    
    ff = FeedForward(d_model=512, d_ff=2048, activation="relu")
    
    assert ff.d_model == 512
    assert ff.d_ff == 2048
    assert ff.activation == "relu"


def test_feed_forward_invalid_d_model():
    """Test that FeedForward raises error for invalid d_model."""
    from adAI.transformer import FeedForward
    
    with pytest.raises(ValueError):
        FeedForward(d_model=0, d_ff=2048)


def test_feed_forward_forward():
    """Test that FeedForward forward pass works correctly."""
    from adAI.transformer import FeedForward
    
    ff = FeedForward(d_model=64, d_ff=128, activation="relu")
    x = np.random.randn(2, 10, 64).astype(np.float32)
    
    output = ff.forward(x)
    
    assert output.shape == (2, 10, 64)


def test_transformer_encoder_layer_creation():
    """Test that TransformerEncoderLayer is created correctly."""
    from adAI.transformer import TransformerEncoderLayer
    
    layer = TransformerEncoderLayer(d_model=512, n_heads=8, d_ff=2048, dropout=0.1)
    
    assert layer.self_attention.d_model == 512
    assert layer.self_attention.n_heads == 8
    assert layer.feed_forward.d_ff == 2048


def test_transformer_encoder_layer_forward():
    """Test that TransformerEncoderLayer forward pass works correctly."""
    from adAI.transformer import TransformerEncoderLayer
    
    layer = TransformerEncoderLayer(d_model=64, n_heads=4, d_ff=128, dropout=0.0)
    x = np.random.randn(2, 10, 64).astype(np.float32)
    
    output = layer.forward(x)
    
    assert output.shape == (2, 10, 64)


def test_generate_transformer_creation():
    """Test that GenerateTransformer creates a Transformer with expected parameters."""
    transformer = GenerateTransformer(
        vocab_size=10000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=128
    )
    
    assert transformer is not None
    assert transformer.vocab_size == 10000
    assert transformer.d_model == 512
    assert transformer.n_heads == 8
    assert transformer.n_layers == 6
    assert transformer.d_ff == 2048
    assert transformer.max_seq_len == 128
    assert len(transformer.layers) == 6


def test_transformer_invalid_vocab_size():
    """Test that Transformer raises error for invalid vocab_size."""
    with pytest.raises(ValueError):
        GenerateTransformer(vocab_size=0, d_model=512)


def test_transformer_invalid_d_model():
    """Test that Transformer raises error for invalid d_model."""
    with pytest.raises(ValueError):
        GenerateTransformer(vocab_size=10000, d_model=0)


def test_transformer_invalid_n_layers():
    """Test that Transformer raises error for invalid n_layers."""
    with pytest.raises(ValueError):
        GenerateTransformer(vocab_size=10000, d_model=512, n_layers=0)


def test_transformer_forward():
    """Test that Transformer forward pass works correctly."""
    transformer = GenerateTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32
    )
    
    X = np.random.randint(0, 1000, (4, 16)).astype(np.int32)
    output = transformer.forward(X)
    
    assert output.shape == (4, 16, 1000)


def test_transformer_train():
    """Test that Transformer training works correctly."""
    transformer = GenerateTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        epochs=5
    )
    
    X_train = np.random.randint(0, 1000, (50, 16)).astype(np.int32)
    y_train = np.random.randint(0, 1000, (50, 16)).astype(np.int32)
    
    history = transformer.train(X_train, y_train, batch_size=16, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 5
    assert history['epochs_trained'] == 5


def test_transformer_train_with_validation():
    """Test that Transformer training with validation works correctly."""
    transformer = GenerateTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        epochs=5
    )
    
    X_train = np.random.randint(0, 1000, (50, 16)).astype(np.int32)
    y_train = np.random.randint(0, 1000, (50, 16)).astype(np.int32)
    X_val = np.random.randint(0, 1000, (10, 16)).astype(np.int32)
    y_val = np.random.randint(0, 1000, (10, 16)).astype(np.int32)
    
    history = transformer.train(X_train, y_train, X_val=X_val, y_val=y_val, batch_size=16, verbose=False)
    
    assert 'val_loss_history' in history
    assert len(history['val_loss_history']) == 5


def test_transformer_predict():
    """Test that Transformer prediction works correctly."""
    transformer = GenerateTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32
    )
    
    X = np.random.randint(0, 1000, (4, 16)).astype(np.int32)
    predictions = transformer.predict(X)
    
    assert predictions.shape == (4, 16)
    assert predictions.dtype == np.int64


def test_transformer_evaluate():
    """Test that Transformer evaluation works correctly."""
    transformer = GenerateTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32
    )
    
    X = np.random.randint(0, 1000, (10, 16)).astype(np.int32)
    y = np.random.randint(0, 1000, (10, 16)).astype(np.int32)
    
    metrics = transformer.evaluate(X, y)
    
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_transformer_save_load():
    """Test that Transformer save and load works correctly."""
    transformer = GenerateTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
        epochs=5
    )
    
    X_train = np.random.randint(0, 1000, (50, 16)).astype(np.int32)
    y_train = np.random.randint(0, 1000, (50, 16)).astype(np.int32)
    transformer.train(X_train, y_train, batch_size=16, verbose=False)
    
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
        temp_path = f.name
    
    try:
        transformer.save(temp_path)
        loaded_transformer = Transformer.load(temp_path)
        
        assert loaded_transformer.vocab_size == transformer.vocab_size
        assert loaded_transformer.d_model == transformer.d_model
        assert loaded_transformer.n_heads == transformer.n_heads
        assert loaded_transformer.n_layers == transformer.n_layers
        assert len(loaded_transformer.loss_history) == len(transformer.loss_history)
    finally:
        import time
        time.sleep(0.1)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def test_transformer_summary():
    """Test that Transformer summary prints without error."""
    transformer = GenerateTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2
    )
    
    # Should not raise any exception
    transformer.summary()
