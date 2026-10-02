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

## CNN agent (pixel observations)

Besides the feature/LIDAR observations, a convolutional agent can be trained on
the rendered game screen. `flappy_bird_gymnasium/envs/pixel_wrapper.py` turns
each frame into a grayscale 84x84 image and stacks the last four frames; the
network (`flappy_bird_gymnasium/tests/cnn_dqn.py`) is a Dueling-DQN with the
Nature-DQN convolution stack, implemented in PyTorch.

Training (Double DQN, writes logs and checkpoints to `runs/cnn/<reward>_seed<seed>/`
and finally exports the best model to `flappy_bird_gymnasium/assets/model/cnn_dqn.pt`):

    $ python -m flappy_bird_gymnasium.train_cnn --steps 1000000
    $ python -m flappy_bird_gymnasium.train_cnn --steps 1000000 --reward risk_averse

`--reward` selects one of the presets in `flappy_bird_gymnasium/rl/rewards.py`
(default `legacy`). The exact reward configuration and all arguments are saved
to `config.json` in the run directory. All hyperparameters are command line
arguments, see `--help`. On an Apple
Silicon Mac (MPS) this runs at roughly 450 steps/s, i.e. 1M steps take about
40 minutes. Use `--no-export` for test runs so they do not overwrite the model
in `assets/model`.

A run directory follows the layout of the shared evaluation (see
*Evaluation*): `train.csv` has one row per training episode with the *score*
(pipes passed) and the *return* (sum of rewards) kept separate, so reward
variants can be compared on the score. `eval.csv` has a greedy measurement (no
exploration, episode seeds `10000 + i` as for every method) every
`--eval-every` steps; the best of these is kept as `best.pt` and is the model
that gets exported. `summary.json` is written last and marks the run as
finished.

The agent trains in the shared environment (`rl.rollout.make_env`): same
reward scheme and frame limit (`--max-episode-steps`, default 3000) as the
other methods. Reaching the limit is a truncation and is bootstrapped, not
treated as a crash. The measurements during training use that limit too, so a
good policy saturates there (`truncation_rate` 1.0); the numbers for a report
come from `summarize`, which plays up to `--eval-max-episode-steps` (20000):

    $ python -m flappy_bird_gymnasium.rl.analysis.summarize runs/cnn --algorithm cnn
    $ python -m flappy_bird_gymnasium.rl.analysis.significance runs/cnn --baseline legacy
    $ python -m flappy_bird_gymnasium.rl.analysis.record runs/cnn/shaped_seed0/best.pt --algorithm cnn --out docs/cnn_shaped.gif

Runs from before the shared evaluation (`log.csv`, no `summary.json`) are
brought into the layout by `backfill`. It only adds files (`train.csv`,
`summary.json`; an old `eval.csv` is kept as `eval_old.csv`). Those runs
trained without a frame limit, so their training curves are not strictly
comparable to new ones, but their checkpoints are:

    $ python -m flappy_bird_gymnasium.evaluate_cnn backfill runs/v2
    $ python -m flappy_bird_gymnasium.rl.analysis.summarize runs/v2 --algorithm cnn

`scripts/run_all.sh` trains every reward preset and seed into `runs/$STUDY/`.

Some options are off by default so that older configurations stay available, but are
worth switching on:

| Flag | Effect |
| --- | --- |
| `--explore-flap-prob 0.08` | Random actions flap with 8 % probability instead of 50 %, i.e. like a trained agent. Combine with `--eps-start 0.3 --eps-decay-steps 200000` so the exploration is actually used. |
| `--n-step 3` | 3-step returns (`R = r_t + γ r_{t+1} + γ² r_{t+2}`, bootstrap with `γ³`). |
| `--buffer-size 250000` | The replay buffer stores one frame per step (~7 KB per transition), so 250k transitions fit in ~1.8 GB. |

Two more groups of options address what the `runs/v2` curves show. In 49 % of
the evaluation points after a run has first passed score 50, the score drops
by more than 40 % below the run's previous best - in every reward preset, so
it is value divergence rather than a reward effect. And four consecutive
frames span only three steps: the bird accelerates by ~0.2 observation pixels
per step², so the curvature of its path is invisible inside such a stack.

**Network and optimiser** (checkpoints stay loadable; `load_cnn()` in
`cnn_dqn.py` reads the variant off the state dict):

| Flag | Effect |
| --- | --- |
| `--layer-norm` | `LayerNorm` before the last ReLU of the trunk. Damps the value divergence behind the post-peak collapses; costs nothing measurable (the conv stack makes 83 % of the FLOPs). |
| `--orthogonal-init` | Orthogonal initialisation (gain √2) of the trunk, gain 1 for the V/A heads. PyTorch's default init is poorly calibrated for ReLU nets. |
| `--adam-eps 1.5e-4` | Adam epsilon as in Atari DQN (torch default 1e-8). Caps the effective step size when gradients get small - the mechanism behind the collapses. |
| `--fc-width 256` | Halves the parameters: 95 % of them sit in the 3136→512 layer, which makes only 17 % of the compute. An ablation of "memory capacity", not a speed knob. |

**Observation** (these change what the network sees: a checkpoint is only
valid for the `--frame-stride` and `--obs-size` it was trained with, and
`play()` / `saliency_cnn` need the same values):

| Flag | Effect |
| --- | --- |
| `--frame-stride 3` | Stack the frames `t, t-3, t-6, t-9` instead of four consecutive ones. The window grows from 3 to 9 steps, so the bird's acceleration becomes visible, while the agent still decides every step (unlike an Atari-style action repeat). |
| `--obs-size 128 84` | Frame height 128 instead of 84. The screen is squeezed 4.8× vertically but only 3.4× horizontally, and the vertical axis carries all the information: a 1-px bird movement is encoded ~50 % more strongly, for ~60 % more FLOPs. |

Watching the trained agent:

```python
from flappy_bird_gymnasium.tests.test_cnn import play
play(epoch=10)               # opens a window; render=False for headless
```

Seeing what the network looks at (Grad-CAM on the last conv layer plus the
input gradient, overlaid on the game frame at several steps of one episode;
`--target value` explains the state value instead of the action choice):

    $ python -m flappy_bird_gymnasium.saliency_cnn --model runs/v2/shaped_seed0/best.pt
