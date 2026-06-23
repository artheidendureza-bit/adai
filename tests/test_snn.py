"""
Unit tests for Spiking Neural Network (SNN) module
"""

import numpy as np
import pytest
from adAI import SNN, GenerateSNN
from adAI.snn import LIFNeuron, SpikingLayer


class TestLIFNeuron:
    """Test LIF neuron functionality"""
    
    def test_lif_initialization(self):
        """Test LIF neuron initialization"""
        neuron = LIFNeuron(threshold=1.0, leak=0.1, reset_mode="subtract")
        assert neuron.threshold == 1.0
        assert neuron.leak == 0.1
        assert neuron.reset_mode == "subtract"
    
    def test_lif_invalid_threshold(self):
        """Test LIF neuron with invalid threshold"""
        with pytest.raises(ValueError):
            LIFNeuron(threshold=-1.0)
    
    def test_lif_invalid_leak(self):
        """Test LIF neuron with invalid leak"""
        with pytest.raises(ValueError):
            LIFNeuron(leak=1.5)
    
    def test_lif_invalid_reset(self):
        """Test LIF neuron with invalid reset mode"""
        with pytest.raises(ValueError):
            LIFNeuron(reset_mode="invalid")
    
    def test_lif_forward_step(self):
        """Test LIF neuron forward step"""
        neuron = LIFNeuron(threshold=1.0, leak=0.1, reset_mode="subtract")
        membrane = np.array([0.5])
        current = np.array([0.6])
        
        new_membrane, spike = neuron.forward_step(membrane, current)
        
        # Membrane should change after adding current
        # After adding current (0.6) to 0.5 = 1.1, it will spike and reset
        # Then decay by 0.1, so should be around 0.0
        assert new_membrane[0] >= 0
        assert spike[0] == True or spike[0] == 1  # Should spike
    
    def test_lif_spike(self):
        """Test LIF neuron spiking"""
        neuron = LIFNeuron(threshold=1.0, leak=0.0, reset_mode="subtract")
        membrane = np.array([0.5])
        current = np.array([1.0])
        
        new_membrane, spike = neuron.forward_step(membrane, current)
        
        # Should spike
        assert spike[0] == True or spike[0] == 1
        # Membrane should be reset
        assert new_membrane[0] < 1.0


class TestSpikingLayer:
    """Test spiking layer functionality"""
    
    def test_spiking_layer_initialization(self):
        """Test spiking layer initialization"""
        layer = SpikingLayer(input_size=4, output_size=8, threshold=1.0, leak=0.1)
        assert layer.input_size == 4
        assert layer.output_size == 8
        assert layer.neuron.threshold == 1.0
    
    def test_spiking_layer_invalid_sizes(self):
        """Test spiking layer with invalid sizes"""
        with pytest.raises(ValueError):
            SpikingLayer(input_size=0, output_size=8)
        with pytest.raises(ValueError):
            SpikingLayer(input_size=4, output_size=0)
    
    def test_spiking_layer_forward_step(self):
        """Test spiking layer forward step"""
        layer = SpikingLayer(input_size=4, output_size=2)
        input_spikes = np.array([[1, 0, 1, 0]], dtype=np.float32)
        
        output_spikes = layer.forward_step(input_spikes)
        
        assert output_spikes.shape == (1, 2)
    
    def test_spiking_layer_reset(self):
        """Test spiking layer reset"""
        layer = SpikingLayer(input_size=4, output_size=2)
        input_spikes = np.array([[1, 0, 1, 0]], dtype=np.float32)
        
        # Run forward to change membrane potential
        layer.forward_step(input_spikes)
        
        # Reset
        layer.reset_state()
        
        # Check membrane is reset
        membrane_np = np.array(layer.membrane_potential)
        assert np.allclose(membrane_np, 0.0)


class TestSNN:
    """Test SNN functionality"""
    
    def test_snn_initialization(self):
        """Test SNN initialization"""
        snn = SNN(
            input_size=10,
            hidden_layers=[32, 16],
            output_size=1,
            time_steps=20,
            reset_mode="subtract"
        )
        assert snn.input_size == 10
        assert snn.hidden_layers == [32, 16]
        assert snn.output_size == 1
        assert snn.time_steps == 20
        assert len(snn.layers) == 3  # 2 hidden + 1 output
    
    def test_snn_invalid_input_size(self):
        """Test SNN with invalid input size"""
        with pytest.raises(ValueError):
            SNN(input_size=0)
    
    def test_snn_invalid_time_steps(self):
        """Test SNN with invalid time steps"""
        with pytest.raises(ValueError):
            SNN(input_size=10, time_steps=0)
    
    def test_snn_invalid_encoding(self):
        """Test SNN with invalid encoding"""
        with pytest.raises(ValueError):
            SNN(input_size=10, encoding="invalid")
    
    def test_snn_forward(self):
        """Test SNN forward pass"""
        snn = SNN(input_size=4, hidden_layers=[8], output_size=1, time_steps=5, reset_mode="subtract")
        X = np.random.randn(10, 4).astype(np.float32)
        
        output = snn.forward(X)
        
        assert output.shape == (10, 1)
    
    def test_snn_predict(self):
        """Test SNN prediction"""
        snn = SNN(input_size=4, hidden_layers=[8], output_size=1, time_steps=5, reset_mode="subtract")
        X = np.random.randn(10, 4).astype(np.float32)
        
        predictions = snn.predict(X)
        
        assert predictions.shape == (10,)
    
    def test_snn_train(self):
        """Test SNN training"""
        snn = SNN(
            input_size=4,
            hidden_layers=[8],
            output_size=1,
            time_steps=5,
            epochs=5,
            learning_rate=0.01,
            reset_mode="subtract"
        )
        X = np.random.randn(50, 4).astype(np.float32)
        y = (X.sum(axis=1) > 0).astype(float)
        
        history = snn.train(X, y, verbose=False)
        
        assert 'loss_history' in history
        assert len(history['loss_history']) == 5
        assert history['epochs_trained'] == 5
    
    def test_snn_train_with_validation(self):
        """Test SNN training with validation"""
        snn = SNN(
            input_size=4,
            hidden_layers=[8],
            output_size=1,
            time_steps=5,
            epochs=5,
            reset_mode="subtract"
        )
        X_train = np.random.randn(50, 4).astype(np.float32)
        y_train = (X_train.sum(axis=1) > 0).astype(float)
        X_val = np.random.randn(10, 4).astype(np.float32)
        y_val = (X_val.sum(axis=1) > 0).astype(float)
        
        history = snn.train(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
        
        assert 'val_loss_history' in history
        assert len(history['val_loss_history']) == 5
    
    def test_snn_early_stopping(self):
        """Test SNN early stopping"""
        snn = SNN(
            input_size=4,
            hidden_layers=[8],
            output_size=1,
            time_steps=5,
            epochs=100,
            reset_mode="subtract"
        )
        X_train = np.random.randn(50, 4).astype(np.float32)
        y_train = (X_train.sum(axis=1) > 0).astype(float)
        X_val = np.random.randn(10, 4).astype(np.float32)
        y_val = (X_val.sum(axis=1) > 0).astype(float)
        
        history = snn.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            early_stopping=True,
            patience=3,
            verbose=False
        )
        
        # Should stop early due to patience
        assert history['epochs_trained'] < 100
    
    def test_snn_evaluate(self):
        """Test SNN evaluation"""
        snn = SNN(input_size=4, hidden_layers=[8], output_size=1, time_steps=5, reset_mode="subtract")
        X = np.random.randn(20, 4).astype(np.float32)
        y = (X.sum(axis=1) > 0).astype(float)
        
        metrics = snn.evaluate(X, y)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert metrics['mse'] >= 0
    
    def test_snn_save_load(self):
        """Test SNN save and load"""
        snn = SNN(
            input_size=4,
            hidden_layers=[8],
            output_size=1,
            time_steps=5,
            epochs=10,
            reset_mode="subtract"
        )
        
        # Train briefly
        X = np.random.randn(20, 4).astype(np.float32)
        y = (X.sum(axis=1) > 0).astype(float)
        snn.train(X, y, verbose=False)
        
        # Save
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            temp_path = f.name
        
        try:
            snn.save(temp_path)
            
            # Load
            loaded_snn = SNN.load(temp_path)
            
            assert loaded_snn.input_size == snn.input_size
            assert loaded_snn.hidden_layers == snn.hidden_layers
            assert loaded_snn.output_size == snn.output_size
            assert loaded_snn.time_steps == snn.time_steps
            assert len(loaded_snn.loss_history) == len(snn.loss_history)
        finally:
            # Wait a bit for file handle to release
            import time
            time.sleep(0.1)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def test_snn_generate_data(self):
        """Test SNN data generation"""
        snn = SNN(input_size=4, hidden_layers=[8], output_size=1, reset_mode="subtract")
        
        data = snn.generate_data(n_train=100, n_val=20, n_test=20)
        
        assert 'X_train' in data
        assert 'y_train' in data
        assert 'X_val' in data
        assert 'y_val' in data
        assert 'X_test' in data
        assert 'y_test' in data
        
        assert data['X_train'].shape == (100, 4)
        assert data['y_train'].shape == (100,)
        assert data['X_val'].shape == (20, 4)
        assert data['y_val'].shape == (20,)
        assert data['X_test'].shape == (20, 4)
        assert data['y_test'].shape == (20,)


class TestGenerateSNN:
    """Test GenerateSNN generator function"""
    
    def test_generate_snn_basic(self):
        """Test basic SNN generation"""
        snn = GenerateSNN(input_size=10, hidden_layers=[32], output_size=1)
        
        assert isinstance(snn, SNN)
        assert snn.input_size == 10
        assert snn.hidden_layers == [32]
        assert snn.output_size == 1
    
    def test_generate_snn_custom_params(self):
        """Test SNN generation with custom parameters"""
        snn = GenerateSNN(
            input_size=5,
            hidden_layers=[16, 8],
            output_size=2,
            learning_rate=0.05,
            epochs=50,
            time_steps=30,
            threshold=0.5,
            leak=0.2,
            reset_mode="zero",
            encoding="poisson"
        )
        
        assert snn.learning_rate == 0.05
        assert snn.epochs == 50
        assert snn.time_steps == 30
        assert snn.threshold == 0.5
        assert snn.leak == 0.2
        assert snn.reset_mode == "zero"
        assert snn.encoding == "poisson"
    
    def test_generate_snn_no_hidden(self):
        """Test SNN generation without hidden layers"""
        snn = GenerateSNN(input_size=4, output_size=1)
        
        assert snn.hidden_layers == []
        assert len(snn.layers) == 1  # Only output layer


class TestSNNEncodings:
    """Test different SNN encoding schemes"""
    
    def test_rate_encoding(self):
        """Test rate encoding"""
        snn = SNN(input_size=4, output_size=1, encoding="rate", time_steps=10, reset_mode="subtract")
        X = np.random.rand(10, 4).astype(np.float32)
        
        output = snn.forward(X)
        
        assert output.shape == (10, 1)
    
    def test_poisson_encoding(self):
        """Test Poisson encoding"""
        snn = SNN(input_size=4, output_size=1, encoding="poisson", time_steps=10, reset_mode="subtract")
        X = np.random.rand(10, 4).astype(np.float32)
        
        output = snn.forward(X)
        
        assert output.shape == (10, 1)


class TestSNNResetModes:
    """Test different reset modes"""
    
    def test_subtract_reset(self):
        """Test subtract reset mode"""
        snn = SNN(input_size=4, output_size=1, reset_mode="subtract", time_steps=5)
        X = np.random.randn(10, 4).astype(np.float32)
        
        output = snn.forward(X)
        
        assert output.shape == (10, 1)
    
    def test_zero_reset(self):
        """Test zero reset mode"""
        snn = SNN(input_size=4, output_size=1, reset_mode="zero", time_steps=5)
        X = np.random.randn(10, 4).astype(np.float32)
        
        output = snn.forward(X)
        
        assert output.shape == (10, 1)
