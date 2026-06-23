from adAI import GeneratePPOAgent, Environment

env = Environment(state_dim=4, action_dim=2, max_steps=100)

agent = GeneratePPOAgent(state_dim=4, action_dim=2, hidden_dims=[64], epochs=50)
agent.train(env, n_episodes=100, verbose=True)

metrics = agent.evaluate(env, n_episodes=10)
print(f"Average Reward: {metrics['average_reward']:.2f}")
