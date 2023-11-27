# A2C RL Agent Visualization for CartPole Environment
<img src="https://gymnasium.farama.org/_images/cart_pole.gif"/>


- This repository provides a simple code snippet for visualizing the behavior of a reinforcement learning (RL) agent trained using the A2C (Advantage Actor-Critic) algorithm on the CartPole environment.

## Getting Started
### Prerequisites
Before running the code, ensure you have the following dependencies installed:

- Gymnasium
- Stable Baselines3

You can install these dependencies using the following:

```bash
pip install gymnasium stable-baselines3
```

## Running the Code
To visualize the RL agent in action, run the provided code snippet. The trained A2C model will perform actions in the CartPole environment, and the interactions will be rendered in a human-friendly format.

```python
import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode="rgb_array")
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
```

## Code Explanation

- Environment Setup: Creates the CartPole environment with rendering enabled.
- Model Training: Initializes and trains the A2C model on the CartPole environment.
- Visualization Loop: Runs a loop to visualize the RL agent's actions in real-time.

## Visualization
The code provides a real-time visualization of the trained A2C model navigating the CartPole environment. This can be useful for understanding the agent's decision-making process.

## Author
Miguel Araújo Julio

## Acknowledgments
Gymnasium and Stable Baselines3 for providing the tools for RL development.
Feel free to customize this readme with your name, additional details, or specific instructions based on your use case.
