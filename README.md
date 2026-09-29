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

The reward function is configurable. The default, `legacy`, reproduces the
scheme of the upstream project:

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

## Evaluation

All learning methods are measured by the same code in
`flappy_bird_gymnasium/rl/analysis/`, so their numbers can sit in one table:
paired episode seeds, an uncensored frame limit, a random-policy reference,
medians with per-seed ranges, and an exact permutation test.

```bash
python -m flappy_bird_gymnasium.rl.analysis.summarize runs/study_reward --algorithm qlearning
python -m flappy_bird_gymnasium.rl.analysis.significance runs/study_reward --baseline legacy
python -m flappy_bird_gymnasium.rl.analysis.plot runs/study_reward --out docs/reward.png
python -m flappy_bird_gymnasium.rl.analysis.plot runs/study_reward --behaviour --out docs/behaviour.png
python -m flappy_bird_gymnasium.rl.analysis.record runs/q1/best.npz --algorithm qlearning --out docs/q1.gif
```

To plug in a method, write its runs in this layout:

```
runs/<study>/<variant>_seed<n>/
    config.json     the run's configuration
    train.csv       one row per training episode, at least `step` and `score`
    eval.csv        optional: periodic greedy measurements (rl/runlog.EVAL_LOG_FIELDS)
    summary.json    written last; marks the run as finished
    <checkpoint>    e.g. best.pt, latest.pt
```

Then add a `RunLoader` that turns a checkpoint into a greedy policy, and give
it one line in `LOADERS` in `rl/analysis/runs.py`.
`flappy_bird_gymnasium/qlearning/evaluate.py` has the example (`RUN_LOADER`).
Until the loader is registered, `--algorithm package.module:RUN_LOADER` works too.

Runs that look different don't have to be rewritten:

* **Other log names** (PPO: `episodes.csv` with `global_step`): set
  `train_log` and `step_column` in the `RunLoader`. For other logs, `plot --runs`
  takes `--source progress.csv --step-column global_step`.
* **Other observations** (CNN: stacked pixel frames): the loader sets
  `LoadedRun.make_env`, built with `rl.rollout.make_env(..., wrapper=...,
  background="night")`, so the reward scheme and frame limit stay the shared ones.
* **Behaviour metrics** (flap rate, offset from the gap centre) are read off the
  game state, so they exist under every observation type.

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

## Reinforcement learning

`flappy_bird_gymnasium.rl` contains a PPO implementation written from scratch in
PyTorch, together with the tooling used to study how the *formulation of the
reward* changes the policy that is learned. It uses the 12-feature observation,
not the LIDAR one.

Install the dependencies first:

    $ pip install -r requirements-dev.txt

### Reward designs

| Preset | Mode | alive | pipe | death | What it asks of the agent |
| --- | --- | --- | --- | --- | --- |
| `legacy` | override | +0.1 | +1.0 | -1.0 | The upstream scheme, unchanged. |
| `additive` | additive | +0.1 | +1.0 | -1.0 | The same terms, summed instead of prioritised. |
| `sparse` | additive | 0 | +1.0 | -1.0 | Pipes and death only; surviving as such earns nothing. |
| `survival` | additive | +0.1 | 0 | -1.0 | Stay alive. Pipes are not rewarded at all. |
| `shaped` | additive | 0 | +1.0 | -1.0 | Adds potential-based shaping towards the gap centre. |
| `risk_averse` | additive | +0.1 | +1.0 | **-5.0** | The same task, but dying costs five times as much. |
| `energy` | additive | +0.1 | +1.0 | -1.0 | Adds a cost of -0.02 per flap. |

`legacy` and `additive` differ only in how the terms combine, which isolates the
effect of the priority chain. `risk_averse` differs from `additive` in a single
number.
