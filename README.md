# gym_multi_env

gym_multi_env is a python module allowing you to join multiple [OpenAI Gym](https://github.com/openai/gym) environments.

## Installation

```shell
$ git clone https://github.com/martinseilair/gym_multi_env/
$ cd gym_multi_env
$ pip install .
```

Tested with Python 3.5.2 and Ubuntu 16.04.

## Quick start

```python
import gym
import gym_multi_env as menv

domain_names = ["CartPole-v0", "Pendulum-v0"]  # list of environments
gym_id = menv.create(domain_names, join_spaces=True)  # create new environment

env = gym.make(gym_id)

for _ in range(1000):
    rgb = env.render(mode="rgb_array")
    obs, reward, done, _ = env.step(env.action_space.sample())  # take a random action

```

## Documentation

### Spaces

If `join_spaces` is set to `False`, observation and action spaces of the chosen environments are joined in a `tuple`.
If `join_spaces` is set to `True` and the types of the spaces are compatible, the spaces will be joined into a space of the same type (with corresponding dimensionality). Otherwise they will be joind in a `tuple`.
To check resulting action and observation space you can use the function `env.info()`.

### Reward

As a default the reward of the multi environment is the sum of the individual environments. 
If you want a particular weighting of the rewards, you can pass a weighting vector with argument `reward_mix`.
Use the function `env.rewards()` to retrieve all individual rewards. 

### Termination

Environment terminates, if at least one individual environment terminates.

### Rendering

Every environment is rendered and shown separately. In `mode="rgb_array"` a `tuple` of the RGB images is put out.

