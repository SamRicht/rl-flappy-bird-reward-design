# RL Flappy Bird — Reward Design

University project investigating **reward functions** in reinforcement learning,
using Flappy Bird as the example environment.

**Maintainers:** Jonathan Künnemann, Colin Hungeling, Max Brinkhoff, Samuel Richter

## Origin

This repository is a fork of
[markub3327/flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium)
(MIT license) by Martin Kubovcik, which in turn builds on
[flappy-bird-gym](https://github.com/Talendar/flappy-bird-gym) by
[@Talendar](https://github.com/Talendar) (Gabriel Nogueira).

The Gymnasium environment, the game logic and the graphics come entirely from
that prior work, see [LICENSE](LICENSE). Our contribution is limited to the
reward functions, the training and the evaluation.

Removed compared to the upstream repository: the pretrained model weights
(`assets/model/*.h5`), the accompanying transformer architecture and the PyPI
deploy workflow. The weights were trained with the original reward function and
are therefore unsuitable for comparing our own reward variants.

## State space

The "FlappyBird-v0" environment, yields simple numerical information about the game's state as
observations representing the game's screen.

### `FlappyBird-v0`
There exist two options for the observations:  
1. option
* The LIDAR sensor 180 readings (Paper: [Playing Flappy Bird Based on Motion Recognition Using a Transformer Model and LIDAR Sensor](https://www.mdpi.com/1424-8220/24/6/1905))

2. option
* the last pipe's horizontal position
* the last top pipe's vertical position
* the last bottom pipe's vertical position
* the next pipe's horizontal position
* the next top pipe's vertical position
* the next bottom pipe's vertical position
* the next next pipe's horizontal position
* the next next top pipe's vertical position
* the next next bottom pipe's vertical position
* player's vertical position
* player's vertical velocity
* player's rotation

## Action space

* 0 - **do nothing**
* 1 - **flap**

## Rewards

The reward is determined in `FlappyBirdEnv.step()`. The order matters: later
assignments **overwrite** earlier ones, they are not summed up.

| Order | Condition | Reward |
|---|---|---|
| 1 | passed a pipe | `+1.0` |
| 2 | LIDAR ray inside the "private zone" (only with `use_lidar=True`) | `-0.5` |
| 3 | otherwise: stayed alive | `+0.1` |
| 4 | bird above the top of the screen (`player_y < 0`) | `-0.5` |
| 5 | collision (terminated) | `-1.0` |

Two consequences of this:

* If the bird passes a pipe and dies in the same frame, the reward is only
  `-1.0`, the `+1.0` from step 1 gets overwritten.
* The proximity penalty in step 2 exists **only in LIDAR mode**. With
  `use_lidar=False` the reward function is a different one, so the two
  observation variants are not directly comparable.

<br>

<p align="center">
  <img align="center" 
       src="https://github.com/markub3327/flappy-bird-gymnasium/blob/main/imgs/dqn.gif?raw=true" 
       width="200"/>
</p>

## Installation

This package is not distributed via PyPI, it is installed locally from the
repository. The `-e` (editable) flag makes changes to the environment take
effect immediately, without reinstalling:

```bash
git clone https://github.com/SamRicht/rl-flappy-bird-reward-design.git
cd rl-flappy-bird-reward-design
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                    # environment + gymnasium, numpy, pygame
pip install -r requirements.txt     # additionally TensorFlow for DQN training
```

## Usage

Like with other `gymnasium` environments, it's very easy to use this package.
Simply import the package and create the environment with the `make` function.
Take a look at the sample code below:

```python
import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()
```

## Playing

To play the game (human mode), run the following command:

    $ flappy_bird_gymnasium
    
To see a random agent playing, add an argument to the command:

    $ flappy_bird_gymnasium --mode random

To see a Deep Q Network agent playing, add an argument to the command:

    $ flappy_bird_gymnasium --mode dqn

> **Note:** `--mode dqn` requires a self-trained checkpoint at
> `flappy_bird_gymnasium/assets/model/dqn.weights.h5`. The upstream pretrained
> weights were removed (see *Origin*); `flappy_bird_gymnasium/tests/test_dqn.py`
> serves as a template for loading your own checkpoint and evaluating it.
