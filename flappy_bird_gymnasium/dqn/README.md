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
python -m flappy_bird_gymnasium.dqn.train --run-name dqn_v2 --reward legacy
```

Ergebnisse landen in `runs/<run-name>/`:

| Datei | Inhalt |
| --- | --- |
| `config.json` | alle Hyperparameter des Laufs, inklusive Reward-Preset |
| `train.csv` | pro Episode: Return, Score, Länge, Flap-Rate, ε, Loss, Q-Mittelwert |
| `eval.csv` | periodische Greedy-Auswertung |
| `best.pt` | Checkpoint mit dem besten Evaluations-Score |
| `latest.pt` | letzter Checkpoint |

Wichtige Flags: `--reward`, `--n-step`, `--learning-rate`, `--batch-size`,
`--gamma`, `--epsilon-decay-steps`, `--target-update-interval`,
`--buffer-size`, `--max-episode-steps`, `--pipe-gap`, `--seed`, sowie
`--no-double` und `--no-dueling` für Ablationsstudien.

## Reward-Schemata

Die Reward-Funktion ist keine Konstante im Spielcode mehr, sondern eine
`RewardConfig` aus [`rl/rewards.py`](../rl/rewards.py) — demselben Modul, das
auch die PPO-Arbeit benutzt. Beide Agenten werden damit gegen exakt dieselben
Reward-Definitionen trainiert und bleiben vergleichbar.

| Preset | Idee |
| --- | --- |
| `legacy` | das Original, Prioritätskette. Baseline des Vergleichs |
| `additive` | dieselben Terme, aber summiert — isoliert den Effekt der Verknüpfung |
| `sparse` | nur Röhre und Tod. Am schwersten zu lernen, aber unverzerrt |
| `survival` | nur Überleben, kein Röhren-Bonus |
| `shaped` | `sparse` plus potential-based Shaping zur Lückenmitte |
| `risk_averse` | Tod kostet −5 statt −1 |
| `energy` | Flattern kostet −0.02 pro Schlag |

```bash
python -m flappy_bird_gymnasium.dqn.train --run-name dqn_sparse --reward sparse
```

## n-step Returns

`--n-step 3` (Standard) speichert statt eines Einzelschritts den Return über
drei Schritte und bootstrappt mit `gamma**3`. Der Grund ist messbar: Der Vogel
braucht ~50 Frames bis zur ersten Röhre, die `+1` muss also bei `n=1` durch 50
Bellman-Updates sickern, bevor sie die erste Aktion der Episode beeinflusst.
`--n-step 1` ergibt das Lehrbuch-DQN zurück.

## Auswerten und Zuschauen

```bash
python -m flappy_bird_gymnasium.dqn.evaluate --checkpoint runs/dqn_v2/best.pt --episodes 50
python -m flappy_bird_gymnasium.dqn.evaluate --checkpoint runs/dqn_v2/best.pt --render --audio
```

Die Umgebung wird aus der im Checkpoint gespeicherten Config rekonstruiert —
ein Agent wird also immer unter dem Reward gemessen, mit dem er trainiert wurde.

## Lernkurven

```bash
python -m flappy_bird_gymnasium.dqn.plot runs/dqn_v2
python -m flappy_bird_gymnasium.dqn.plot runs/dqn_legacy runs/dqn_sparse --out vergleich.png
```

Sechs Panels: Score, Greedy-Evaluation, Return, Episodenlänge, TD-Loss und
Flap-Rate. Beim Vergleich verschiedener Reward-Schemata ist **nur der Score**
aussagekräftig — Returns verschiedener Schemata sind per Konstruktion nicht
vergleichbar.

## Tests

```bash
python -m pytest flappy_bird_gymnasium/tests/test_dqn_agent.py -q
```

## Aufbau

| Modul | Aufgabe |
| --- | --- |
| `config.py` | `DQNConfig` — sämtliche Hyperparameter als Dataclass |
| `replay_buffer.py` | Ringpuffer plus `NStepAccumulator` für n-step Returns |
| `model.py` | `QNetwork` (MLP) und `DuelingQNetwork` (V/A-Zerlegung) |
| `agent.py` | ε-greedy Aktionswahl, Bellman-Target, Gradientenschritt, Speichern/Laden |
| `env_utils.py` | `make_env` inklusive Reward-Preset und TimeLimit, Seeding |
| `train.py` | Trainingsloop, CSV-Logging, Checkpoints, periodische Evaluation |
| `evaluate.py` | Greedy-Durchläufe, optional mit Fenster |
| `plot.py` | Lernkurven, auch für den Vergleich mehrerer Läufe |

Die Reward-Definitionen liegen bewusst **nicht** hier, sondern in
[`rl/rewards.py`](../rl/rewards.py), gemeinsam mit der PPO-Arbeit.

## Drei Details, die leicht schiefgehen

**`terminated` ≠ `truncated`.** Nur `terminated` (der Vogel ist abgestürzt)
beendet das Bootstrapping in der Bellman-Gleichung. `truncated` bedeutet, dass
das Schrittlimit erreicht wurde — die Episode endet, aber die Zukunft war nicht
wertlos. Würde man das als terminal speichern, lernte der Agent, dass Erfolg
wie Sterben aussieht. Deshalb hat der n-step-Akkumulator zwei getrennte Wege:
`push(..., terminated=True)` leert die Queue als Tod, `flush()` leert sie mit
erhaltenem Bootstrap.

**Jede Transition trägt ihren eigenen Diskont.** Am Episodenende werden die
noch wartenden Schritte mit *kürzerem* Horizont geleert. Ein globales
`gamma**n` wäre für die falsch, deshalb speichert der Puffer `gamma**horizon`
pro Transition.

**Huber statt MSE.** Der Reward von +0.1 pro Frame summiert sich bei γ=0.99 zu
großen Returns auf. MSE plus große TD-Fehler ergibt Gradienten, die das Netz
zerlegen; `smooth_l1_loss` und Gradient-Clipping halten das stabil.

## Nächste Ausbaustufen

- Prioritized Experience Replay
- LIDAR-Beobachtungen (`use_lidar=True`, 180 Dimensionen) mit Frame-Stacking
- Anbindung an `rl/plotting.py` und `rl/summarize.py`, sobald der PPO-Branch
  gemergt ist — dann liegen DQN- und PPO-Ergebnisse in derselben Auswertung
