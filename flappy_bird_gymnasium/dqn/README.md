# DQN-Agent für FlappyBird-v0

PyTorch-Implementierung eines Deep Q-Networks für die Gymnasium-Umgebung
`FlappyBird-v0` — mit Erfahrungsspeicher, Target-Netzwerk, Double DQN und
Dueling-Kopf.

## Installation

Die venv liegt **neben** dem Repo, nicht darin — also unter
`Flappy-Bird-KI\.venv`, während das Repo `Flappy-Bird-KI\rl-flappy-bird-reward-design`
ist. Neu aufsetzen:

```bash
cd Flappy-Bird-KI
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e rl-flappy-bird-reward-design
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## venv benutzen

Alle Befehle unten laufen aus dem Repo-Verzeichnis. Entweder die venv einmal
pro Terminal aktivieren:

```powershell
cd C:\Users\1dv95\Flappy-Bird-KI\rl-flappy-bird-reward-design
..\.venv\Scripts\Activate.ps1      # PowerShell
..\.venv\Scripts\activate.bat      # cmd.exe
```

…oder das venv-Python direkt ansprechen, ohne zu aktivieren:

```powershell
..\.venv\Scripts\python.exe -m flappy_bird_gymnasium.dqn.train ...
```

Ein blankes `python` benutzt das System-Python, in dem das Paket **nicht**
installiert ist — das ergibt `ModuleNotFoundError: No module named
'flappy_bird_gymnasium'`.

## Training

```bash
python -m flappy_bird_gymnasium.dqn.train --run-name dqn_v1 --total-steps 500000
```

Ergebnisse landen in `runs/<run-name>/`:

| Datei | Inhalt |
| --- | --- |
| `config.json` | alle Hyperparameter des Laufs |
| `train.csv` | pro Episode: Return, Score, Länge, ε, Loss, Q-Mittelwert |
| `eval.csv` | periodische Greedy-Auswertung |
| `best.pt` | Checkpoint mit dem besten Evaluations-Score |
| `latest.pt` | letzter Checkpoint |

Wichtige Flags: `--learning-rate`, `--batch-size`, `--gamma`,
`--epsilon-decay-steps`, `--target-update-interval`, `--buffer-size`,
`--seed`, sowie `--no-double` und `--no-dueling` für Ablationsstudien.

## Auswerten und Zuschauen

```bash
python -m flappy_bird_gymnasium.dqn.evaluate --checkpoint runs/dqn_v1/best.pt --episodes 30
python -m flappy_bird_gymnasium.dqn.evaluate --checkpoint runs/dqn_v1/best.pt --render --audio
```

## Lernkurven

```bash
python -m flappy_bird_gymnasium.dqn.plot runs/dqn_v1
python -m flappy_bird_gymnasium.dqn.plot runs/dqn_v1 runs/dqn_no_double --out vergleich.png
```

## Aufbau

| Modul | Aufgabe |
| --- | --- |
| `config.py` | `DQNConfig` — sämtliche Hyperparameter als Dataclass |
| `replay_buffer.py` | uniformer Ringpuffer auf vorallokierten numpy-Arrays |
| `model.py` | `QNetwork` (MLP) und `DuelingQNetwork` (V/A-Zerlegung) |
| `agent.py` | ε-greedy Aktionswahl, Bellman-Target, Gradientenschritt, Speichern/Laden |
| `env_utils.py` | `make_env`, Seeding |
| `train.py` | Trainingsloop, CSV-Logging, Checkpoints, periodische Evaluation |
| `evaluate.py` | Greedy-Durchläufe, optional mit Fenster |
| `plot.py` | Lernkurven, auch für den Vergleich mehrerer Läufe |

## Zwei Details, die leicht schiefgehen

**`terminated` ≠ `truncated`.** Nur `terminated` (der Vogel ist abgestürzt)
beendet das Bootstrapping in der Bellman-Gleichung. `truncated` bedeutet, dass
das `score_limit` erreicht wurde — die Episode endet, aber die Zukunft war
nicht wertlos. Würde man das als terminal speichern, lernte der Agent, dass
Erfolg wie Sterben aussieht.

**Huber statt MSE.** Der Reward von +0.1 pro Frame summiert sich bei γ=0.99 zu
großen Returns auf. MSE plus große TD-Fehler ergibt Gradienten, die das Netz
zerlegen; `smooth_l1_loss` und Gradient-Clipping halten das stabil.

## Nächste Ausbaustufen

- n-step Returns (n = 3) — schnellere Belohnungsweitergabe
- Prioritized Experience Replay
- Reward-Design-Experimente (Thema des Repos): z. B. ein Shaping-Term für den
  Abstand zur Pipe-Mitte, Ablation des `+0.1`-Alive-Rewards
- LIDAR-Beobachtungen (`use_lidar=True`, 180 Dimensionen) mit Frame-Stacking
