"""
Unit tests for the Reinforcement Learning module of adAI
"""

import pytest
import numpy as np
from adAI import GeneratePPOAgent, PPOAgent, Environment


def test_environment_creation():
    """Test that Environment is created with correct parameters."""
    env = Environment(state_dim=4, action_dim=2, max_steps=100)
    
    assert env.state_dim == 4
    assert env.action_dim == 2
    assert env.max_steps == 100


def test_environment_invalid_state_dim():
    """Test that Environment raises error for invalid state_dim."""
    with pytest.raises(ValueError):
        Environment(state_dim=0, action_dim=2)


def test_environment_invalid_action_dim():
    """Test that Environment raises error for invalid action_dim."""
    with pytest.raises(ValueError):
        Environment(state_dim=4, action_dim=0)


def test_environment_reset():
    """Test that Environment reset works correctly."""
    env = Environment(state_dim=4, action_dim=2)
    state = env.reset()
    
    assert state.shape == (4,)
    assert env.current_step == 0


def test_environment_step():
    """Test that Environment step works correctly."""
    env = Environment(state_dim=4, action_dim=2, max_steps=10)
    env.reset()
    
    next_state, reward, done, info = env.step(0)
    
    assert next_state.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert env.current_step == 1


def test_environment_episode_end():
    """Test that Environment episode ends after max_steps."""
    env = Environment(state_dim=4, action_dim=2, max_steps=5)
    env.reset()
    
    for _ in range(5):
        next_state, reward, done, info = env.step(0)
    
    assert done == True


def test_policy_network_creation():
    """Test that PolicyNetwork is created with correct parameters."""
    from adAI.rl import PolicyNetwork
    
    policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dims=[16, 8])
    
    assert policy.state_dim == 4
    assert policy.action_dim == 2
    assert policy.hidden_dims == [16, 8]


def test_policy_network_invalid_state_dim():
    """Test that PolicyNetwork raises error for invalid state_dim."""
    from adAI.rl import PolicyNetwork
    
    with pytest.raises(ValueError):
        PolicyNetwork(state_dim=0, action_dim=2)


def test_policy_network_invalid_action_dim():
    """Test that PolicyNetwork raises error for invalid action_dim."""
    from adAI.rl import PolicyNetwork
    
    with pytest.raises(ValueError):
        PolicyNetwork(state_dim=4, action_dim=0)


def test_policy_network_forward():
    """Test that PolicyNetwork forward pass works correctly."""
    from adAI.rl import PolicyNetwork
    
    policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dims=[16])
    state = np.random.randn(3, 4).astype(np.float32)
    
    action_probs = policy.forward(state)
    
    assert action_probs.shape == (3, 2)
    assert np.allclose(action_probs.sum(axis=1), 1.0, atol=1e-6)


def test_policy_network_sample_action():
    """Test that PolicyNetwork sample action works correctly."""
    from adAI.rl import PolicyNetwork
    
    policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dims=[16])
    state = np.random.randn(4).astype(np.float32)
    
    action = policy.sample_action(state)
    
    assert action in [0, 1]
    assert isinstance(action, int)


def test_value_network_creation():
    """Test that ValueNetwork is created with correct parameters."""
    from adAI.rl import ValueNetwork
    
    value = ValueNetwork(state_dim=4, hidden_dims=[16, 8])
    
    assert value.state_dim == 4
    assert value.hidden_dims == [16, 8]


def test_value_network_invalid_state_dim():
    """Test that ValueNetwork raises error for invalid state_dim."""
    from adAI.rl import ValueNetwork
    
    with pytest.raises(ValueError):
        ValueNetwork(state_dim=0)


def test_value_network_forward():
    """Test that ValueNetwork forward pass works correctly."""
    from adAI.rl import ValueNetwork
    
    value = ValueNetwork(state_dim=4, hidden_dims=[16])
    state = np.random.randn(3, 4).astype(np.float32)
    
    values = value.forward(state)
    
    assert values.shape == (3, 1)


def test_generate_ppo_agent_creation():
    """Test that GeneratePPOAgent creates a PPOAgent with expected parameters."""
    agent = GeneratePPOAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 32],
        gamma=0.99
    )
    
    assert agent is not None
    assert agent.state_dim == 4
    assert agent.action_dim == 2
    assert agent.hidden_dims == [64, 32]
    assert agent.gamma == 0.99
    assert agent.epsilon == 0.2


def test_ppo_agent_invalid_state_dim():
    """Test that PPOAgent raises error for invalid state_dim."""
    with pytest.raises(ValueError):
        GeneratePPOAgent(state_dim=0, action_dim=2)


def test_ppo_agent_invalid_action_dim():
    """Test that PPOAgent raises error for invalid action_dim."""
    with pytest.raises(ValueError):
        GeneratePPOAgent(state_dim=4, action_dim=0)


def test_ppo_agent_invalid_gamma():
    """Test that PPOAgent raises error for invalid gamma."""
    with pytest.raises(ValueError):
        GeneratePPOAgent(state_dim=4, action_dim=2, gamma=1.5)


def test_ppo_agent_summary():
    """Test that PPOAgent summary prints without error."""
    agent = GeneratePPOAgent(state_dim=4, action_dim=2, hidden_dims=[64, 32])
    
    # Should not raise any exception
    agent.summary()


def test_ppo_agent_forward():
    """Test that PPOAgent forward pass works correctly."""
    agent = GeneratePPOAgent(state_dim=4, action_dim=2, hidden_dims=[64])
    state = np.random.randn(3, 4).astype(np.float32)
    
    output = agent.forward(state)
    
    assert output.shape == (3, 2)


def test_ppo_agent_collect_episode():
    """Test that PPOAgent episode collection works correctly."""
    env = Environment(state_dim=4, action_dim=2, max_steps=10)
    agent = GeneratePPOAgent(state_dim=4, action_dim=2, hidden_dims=[64])
    
    trajectory = agent.collect_episode(env, max_steps=5)
    
    assert len(trajectory) > 0
    assert len(trajectory) <= 5
    assert len(trajectory[0]) == 5  # (state, action, reward, next_state, done)


def test_ppo_agent_train():
    """Test that PPOAgent training works correctly."""
    env = Environment(state_dim=4, action_dim=2, max_steps=10)
    agent = GeneratePPOAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64],
        epochs=5
    )
    
    history = agent.train(env, n_episodes=20, batch_size=4, verbose=False)
    
    assert 'loss_history' in history
    assert len(history['loss_history']) == 20
    assert history['epochs_trained'] == 20


def test_ppo_agent_predict():
    """Test that PPOAgent prediction works correctly."""
    agent = GeneratePPOAgent(state_dim=4, action_dim=2, hidden_dims=[64])
    state = np.random.randn(4).astype(np.float32)
    
    action = agent.predict(state)
    
    assert action in [0, 1]
    assert isinstance(action, int)


def test_ppo_agent_evaluate():
    """Test that PPOAgent evaluation works correctly."""
    env = Environment(state_dim=4, action_dim=2, max_steps=10)
    agent = GeneratePPOAgent(state_dim=4, action_dim=2, hidden_dims=[64])
    
    metrics = agent.evaluate(env, n_episodes=5)
    
    assert 'average_reward' in metrics
    assert isinstance(metrics['average_reward'], float)


def test_ppo_agent_save_load():
    """Test that PPOAgent save and load works correctly."""
    env = Environment(state_dim=4, action_dim=2, max_steps=10)
    agent = GeneratePPOAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64],
        epochs=5
    )
    
    agent.train(env, n_episodes=10, batch_size=4, verbose=False)
    
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
        temp_path = f.name
    
    try:
        agent.save(temp_path)
        loaded_agent = PPOAgent.load(temp_path)
        
        assert loaded_agent.state_dim == agent.state_dim
        assert loaded_agent.action_dim == agent.action_dim
        assert loaded_agent.hidden_dims == agent.hidden_dims
        assert loaded_agent.gamma == agent.gamma
        assert len(loaded_agent.loss_history) == len(agent.loss_history)
    finally:
        import time
        time.sleep(0.1)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def test_ppo_agent_custom_gamma():
    """Test that custom gamma works correctly."""
    agent = GeneratePPOAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64],
        gamma=0.95
    )
    
    assert agent.gamma == 0.95


def test_ppo_agent_custom_epsilon():
    """Test that custom epsilon works correctly."""
    agent = GeneratePPOAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64],
        epsilon=0.1
    )
    
    assert agent.epsilon == 0.1


def test_ppo_agent_early_stopping():
    """Test that PPOAgent early stopping works correctly."""
    env = Environment(state_dim=4, action_dim=2, max_steps=10)
    agent = GeneratePPOAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64],
        epochs=100
    )
    
    history = agent.train(
        env,
        n_episodes=50,
        batch_size=4,
        early_stopping=True,
        patience=5,
        verbose=False
    )
    
    # Should stop early due to patience
    assert history['epochs_trained'] <= 50
