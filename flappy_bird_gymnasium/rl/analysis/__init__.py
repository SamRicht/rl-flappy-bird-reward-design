"""Evaluation chain shared by every learning method: measure, test, draw, record.

One implementation for all four methods, so that a number from the DQN work
and a number from the tabular agent were produced the same way and can sit in
one table. Four stages, each writing files the next one reads::

    python -m flappy_bird_gymnasium.rl.analysis.summarize runs/study_reward \\
        --algorithm qlearning
    python -m flappy_bird_gymnasium.rl.analysis.significance runs/study_reward \\
        --baseline legacy
    python -m flappy_bird_gymnasium.rl.analysis.plot runs/study_reward \\
        --out docs/reward.png
    python -m flappy_bird_gymnasium.rl.analysis.record runs/q1/best.npz \\
        --algorithm qlearning --out docs/q1.gif

The chain knows nothing about any method. What it relies on is the layout of a
run directory, the same one the DQN and tabular packages already write::

    <study>/<variant>_seed<n>/
        config.json    the run's configuration
        train.csv      one row per training episode, at least ``step, score``
        eval.csv       optional, periodic greedy measurements (``EVAL_LOG_FIELDS``)
        summary.json   written last; its presence marks the run as finished
        <checkpoint>   whatever the method saves, e.g. ``best.npz``, ``best.pt``

and one :class:`~flappy_bird_gymnasium.rl.analysis.runs.RunLoader` per method
that turns a checkpoint into a policy. Registering a method is one line in
:data:`~flappy_bird_gymnasium.rl.analysis.runs.LOADERS`. A method whose
training log has another name or step column says so in its loader, and one
that sees the game through other observations brings its own environment via
``LoadedRun.make_env``.
"""
