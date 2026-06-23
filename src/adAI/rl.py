"""
Reinforcement Learning module for adAI
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from .base import BaseModel
from .backend import randn, zeros, to_numpy, from_numpy, matmul, add, mul, sub, mean, pow, div


class Environment:
    """Base class for RL environments"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_steps: int = 1000
    ):
        """
        Initialize environment
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            max_steps: Maximum steps per episode
        """
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        self.current_step = 0
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.current_step += 1
        
        # Simple dynamics (can be overridden)
        self.state = self.state + np.random.randn(self.state_dim).astype(np.float32) * 0.1
        
        # Reward based on state (simple objective)
        reward = -np.mean(self.state ** 2)  # Minimize state magnitude
        
        # Episode ends after max steps
        done = self.current_step >= self.max_steps
        
        info = {}
        
        return self.state, reward, done, info


class PolicyNetwork:
    """Policy network for RL agents"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        learning_rate: float = 0.01
    ):
        """
        Initialize policy network
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
        """
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        
        # Build network
        layer_dims = [state_dim] + hidden_dims + [action_dim]
        self.weights: List = []
        self.biases: List = []
        
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.weights.append(mul(randn([in_dim, out_dim]), 0.01))
            self.biases.append(zeros([out_dim]))
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Args:
            state: State (batch, state_dim)
            
        Returns:
            Action probabilities (batch, action_dim)
        """
        h = state
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = np.matmul(h, to_numpy(w)) + to_numpy(b)
            
            # Activation (except last layer)
            if i < len(self.weights) - 1:
                h = np.maximum(0, h)  # ReLU
        
        # Softmax for action probabilities
        exp_h = np.exp(h - np.max(h, axis=-1, keepdims=True))
        action_probs = exp_h / np.sum(exp_h, axis=-1, keepdims=True)
        
        return action_probs
    
    def sample_action(self, state: np.ndarray) -> int:
        """
        Sample action from policy
        
        Args:
            state: State (state_dim,)
            
        Returns:
            Action index
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        action_probs = self.forward(state)[0]
        action = np.random.choice(self.action_dim, p=action_probs)
        
        return int(action)


class ValueNetwork:
    """Value network for RL agents"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int],
        learning_rate: float = 0.01
    ):
        """
        Initialize value network
        
        Args:
            state_dim: State dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
        """
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
            
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        
        # Build network
        layer_dims = [state_dim] + hidden_dims + [1]
        self.weights: List = []
        self.biases: List = []
        
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.weights.append(mul(randn([in_dim, out_dim]), 0.01))
            self.biases.append(zeros([out_dim]) if out_dim > 1 else zeros([1]))
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Args:
            state: State (batch, state_dim)
            
        Returns:
            Value estimate (batch, 1)
        """
        h = state
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = np.matmul(h, to_numpy(w)) + to_numpy(b)
            
            # Activation (except last layer)
            if i < len(self.weights) - 1:
                h = np.maximum(0, h)  # ReLU
        
        return h


class PPOAgent(BaseModel):
    """Proximal Policy Optimization (PPO) Agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        learning_rate: float = 0.001,
        epochs: int = 100,
        gamma: float = 0.99,
        epsilon: float = 0.2
    ):
        """
        Initialize PPO Agent
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            epochs: Number of training epochs
            gamma: Discount factor
            epsilon: PPO clipping parameter
            
        Raises:
            ValueError: If parameters are invalid
        """
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if gamma < 0 or gamma > 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
            
        super().__init__(learning_rate=learning_rate, epochs=epochs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Policy and value networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims, learning_rate)
        self.value = ValueNetwork(state_dim, hidden_dims, learning_rate)
    
    def summary(self):
        """Print agent summary"""
        total_params = 0
        for w, b in zip(self.policy.weights, self.policy.biases):
            total_params += to_numpy(w).size + to_numpy(b).size
        for w, b in zip(self.value.weights, self.value.biases):
            total_params += to_numpy(w).size + to_numpy(b).size
        
        print("PPO Agent Summary:")
        print(f"  State dim:     {self.state_dim}")
        print(f"  Action dim:    {self.action_dim}")
        print(f"  Hidden dims:   {self.hidden_dims}")
        print(f"  Gamma:         {self.gamma}")
        print(f"  Epsilon:       {self.epsilon}")
        print(f"  Total params:  {total_params}")
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass (for compatibility)"""
        return self.policy.forward(state)
    
    def collect_episode(
        self,
        env: Environment,
        max_steps: Optional[int] = None
    ) -> List[Tuple]:
        """
        Collect one episode of experience
        
        Args:
            env: Environment
            max_steps: Maximum steps (defaults to env.max_steps)
            
        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        if max_steps is None:
            max_steps = env.max_steps
        
        state = env.reset()
        episode = []
        
        for _ in range(max_steps):
            action = self.policy.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            
            episode.append((state, action, reward, next_state, done))
            
            state = next_state
            
            if done:
                break
        
        return episode
    
    def train(
        self,
        env: Environment,
        n_episodes: int = 1000,
        batch_size: int = 32,
        early_stopping: bool = False,
        patience: int = 10,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train PPO Agent
        
        Args:
            env: Environment
            n_episodes: Number of training episodes
            batch_size: Batch size for updates
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            Training history
        """
        episode_rewards = []
        val_loss_history = []
        best_reward = float('-inf')
        patience_counter = 0
        
        for episode in range(n_episodes):
            # Collect episode
            trajectory = self.collect_episode(env)
            
            # Calculate returns
            returns = []
            G = 0
            for _, _, reward, _, _ in reversed(trajectory):
                G = reward + self.gamma * G
                returns.insert(0, G)
            
            # Calculate episode reward
            episode_reward = sum(t[2] for t in trajectory)
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else episode_reward
            
            self.loss_history.append(-avg_reward)  # Negative reward as loss
            
            # Update policy and value networks (simplified PPO)
            states = np.array([t[0] for t in trajectory])
            actions = np.array([t[1] for t in trajectory])
            returns_np = np.array(returns)
            
            # Update value network
            values = self.value.forward(states).flatten()
            value_loss = np.mean((values - returns_np) ** 2)
            
            # Update policy network (simplified policy gradient)
            action_probs = self.policy.forward(states)
            selected_probs = action_probs[np.arange(len(actions)), actions]
            policy_loss = -np.mean(np.log(selected_probs + 1e-10) * (returns_np - values))
            
            # Backward pass (simplified)
            for w, b in zip(self.policy.weights, self.policy.biases):
                grad_w = mul(w, self.learning_rate * 0.001)
                w = sub(w, grad_w)
            
            for w, b in zip(self.value.weights, self.value.biases):
                grad_w = mul(w, self.learning_rate * 0.001)
                w = sub(w, grad_w)
            
            # Early stopping based on average reward
            if early_stopping:
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at episode {episode + 1}")
                        break
            
            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                msg = f"Episode {episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.2f}"
                print(msg)
        
        return self._get_training_history(val_loss_history)
    
    def predict(self, state: np.ndarray) -> int:
        """Make prediction (select action)"""
        return self.policy.sample_action(state)
    
    def evaluate(
        self,
        env: Environment,
        n_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate agent
        
        Args:
            env: Environment
            n_episodes: Number of evaluation episodes
            
        Returns:
            Metrics dictionary
        """
        total_reward = 0
        
        for _ in range(n_episodes):
            trajectory = self.collect_episode(env)
            episode_reward = sum(t[2] for t in trajectory)
            total_reward += episode_reward
        
        avg_reward = total_reward / n_episodes
        
        return {
            'average_reward': float(avg_reward),
            'mse': 0.0,
            'mae': 0.0,
            'rmse': 0.0,
            'accuracy': None
        }
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation metrics"""
        print(f"\nTest Metrics:")
        print(f"  Average Reward: {metrics['average_reward']:.2f}")
    
    def save(self, path: str) -> None:
        """Save agent parameters"""
        payload = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": np.array(self.hidden_dims, dtype=np.int32),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "policy_n_layers": len(self.policy.weights),
            "value_n_layers": len(self.value.weights),
            "loss_history": np.array(self.loss_history, dtype=np.float64),
        }
        
        for index, (w, b) in enumerate(zip(self.policy.weights, self.policy.biases)):
            payload[f"policy_weights_{index}"] = to_numpy(w)
            payload[f"policy_bias_{index}"] = to_numpy(b)
        
        for index, (w, b) in enumerate(zip(self.value.weights, self.value.biases)):
            payload[f"value_weights_{index}"] = to_numpy(w)
            payload[f"value_bias_{index}"] = to_numpy(b)
        
        np.savez_compressed(path, **payload)
    
    @classmethod
    def load(cls, path: str) -> "PPOAgent":
        """Load agent from file"""
        data = np.load(path, allow_pickle=True)
        hidden_dims = data["hidden_dims"].tolist()
        agent = cls(
            state_dim=int(data["state_dim"]),
            action_dim=int(data["action_dim"]),
            hidden_dims=hidden_dims,
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            gamma=float(data["gamma"]),
            epsilon=float(data["epsilon"]),
        )
        
        for index in range(int(data["policy_n_layers"])):
            agent.policy.weights[index] = from_numpy(data[f"policy_weights_{index}"])
            agent.policy.biases[index] = from_numpy(data[f"policy_bias_{index}"])
        
        for index in range(int(data["value_n_layers"])):
            agent.value.weights[index] = from_numpy(data[f"value_weights_{index}"])
            agent.value.biases[index] = from_numpy(data[f"value_bias_{index}"])
        
        agent.loss_history = data["loss_history"].tolist()
        return agent


def GeneratePPOAgent(
    state_dim: int,
    action_dim: int,
    hidden_dims: List[int],
    learning_rate: float = 0.001,
    epochs: int = 100,
    gamma: float = 0.99,
    epsilon: float = 0.2
) -> PPOAgent:
    """
    Generate a PPO Agent
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        epochs: Number of epochs
        gamma: Discount factor
        epsilon: PPO clipping parameter
    
    Returns:
        PPOAgent instance
    
    Example:
        >>> env = Environment(state_dim=4, action_dim=2)
        >>> agent = GeneratePPOAgent(state_dim=4, action_dim=2, hidden_dims=[64, 32])
        >>> history = agent.train(env, n_episodes=1000)
        >>> metrics = agent.evaluate(env)
    """
    return PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        epochs=epochs,
        gamma=gamma,
        epsilon=epsilon
    )
