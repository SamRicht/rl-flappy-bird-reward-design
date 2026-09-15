# DQN-Agent für FlappyBird-v0

PyTorch-Implementierung eines Deep Q-Networks für die Gymnasium-Umgebung
`FlappyBird-v0` — mit Erfahrungsspeicher, Target-Netzwerk, Double DQN,
Dueling-Kopf und n-step Returns. Dazu ein Runner für Studien über mehrere Seeds
und eine Auswertung, deren Zahlen auch neben PPO, Q-Learning und CNN bestehen.

> **[ARBEITSSTAND.md](ARBEITSSTAND.md)** dokumentiert den vollständigen Stand:
> alle Entscheidungen mit Begründung, die gelaufenen Experimente mit Zahlen, die
> gefundenen Fehler, die Methodik für den Vier-Wege-Vergleich und die Hypothesen
> zu den geplanten Läufen. Diese Datei hier ist die Bedienungsanleitung.

## venv benutzen

Die venv liegt **neben** dem Repo, nicht darin — also unter
`Flappy-Bird-KI\.venv`, während das Repo `Flappy-Bird-KI\rl-flappy-bird-reward-design`
ist. Alle Befehle unten laufen aus dem Repo-Verzeichnis:

```powershell
cd C:\Users\...\Flappy-Bird-KI\rl-flappy-bird-reward-design
..\.venv\Scripts\Activate.ps1      # PowerShell
..\.venv\Scripts\activate.bat      # cmd.exe
```

…oder ohne Aktivieren das venv-Python direkt ansprechen:

```powershell
..\.venv\Scripts\python.exe -m flappy_bird_gymnasium.dqn.train ...
```

Ein blankes `python` benutzt das System-Python, in dem das Paket **nicht**
installiert ist — das ergibt `ModuleNotFoundError: No module named
'flappy_bird_gymnasium'`.

Neu aufsetzen:

```bash
cd Flappy-Bird-KI
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e rl-flappy-bird-reward-design
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Einzelner Lauf

```bash
python -m flappy_bird_gymnasium.dqn.train --run-name dqn_v3 --reward legacy
```

Ergebnisse landen in `runs/<run-name>/`:

| Datei | Inhalt |
| --- | --- |
| `config.json` | alle Hyperparameter des Laufs |
| `train.csv` | pro Episode: Return, Score, Länge, Flap-Rate, ε, Loss, Q-Mittelwert |
| `eval.csv` | periodische Greedy-Auswertung inklusive `truncation_rate` |
| `summary.json` | Kennzahlen des Laufs, inklusive `steps_to_<n>` |
| `best.pt` / `latest.pt` | bester und letzter Checkpoint |

## Studien über mehrere Seeds

**Ein einzelner Lauf beweist nichts.** Die `dqn_v2`-Baseline schwankte zwischen
zwei Auswertungspunkten 50.000 Schritte auseinander von Score 69,4 auf 1,4. Wer
zwei Varianten mit je einem Seed vergleicht, vergleicht Glück. Deshalb läuft
jede Aussage über mehrere Seeds:

```bash
# Wie stark streut der Algorithmus überhaupt? Der Maßstab für alles andere.
python -m flappy_bird_gymnasium.dqn.experiments seeds --seeds 8 --total-steps 1000000

# Was trägt jeder Baustein bei?
python -m flappy_bird_gymnasium.dqn.experiments ablation --seeds 3

# Ein Parameter über mehrere Werte
python -m flappy_bird_gymnasium.dqn.experiments sweep --param learning_rate \
    --values 5e-5 1e-4 3e-4 --seeds 3

# Reward-Schemata gegeneinander
python -m flappy_bird_gymnasium.dqn.experiments reward --seeds 3
```

Die Läufe gehen in einen Prozess-Pool, jeder Worker mit genau einem
Torch-Thread — sonst konkurrieren die Läufe um Kerne und jeder einzelne wird
langsamer, als er allein wäre. `--workers 0` bedeutet Kernzahl minus zwei.

### Ablationsvarianten

| Variante | abgeschaltet |
| --- | --- |
| `full` | nichts — Double DQN, Dueling und n-step 3 |
| `no_double` | Double DQN |
| `no_dueling` | Dueling-Kopf |
| `no_nstep` | n-step (n = 1) |
| `vanilla` | alle drei — DQN wie ursprünglich veröffentlicht |

### Auswertung einer Studie

```bash
python -m flappy_bird_gymnasium.dqn.summarize runs/study_ablation --episodes 30
```

Schreibt `evaluations.csv` (pro Lauf) und `aggregate.csv` (pro Variante) und
gibt eine sortierte Tabelle aus. Alle Läufe werden auf **denselben**
Evaluations-Seeds gemessen, sehen also identische Röhrenfolgen — der Vergleich
ist gepaart und nicht davon verfälscht, wer die leichteren Level gezogen hat.

## Welche Zahl wofür

Für den Vergleich mehrerer Lernverfahren ist die Wahl der Metrik die halbe
Miete. Drei Größen, drei Zwecke:

| Metrik | Aussage | Grenze |
| --- | --- | --- |
| **Score** (passierte Röhren) | was die fertige Policy kann | wird vom Frame-Limit zensiert |
| **`steps_to_<n>`** | wie schnell gelernt wurde | nicht zensierbar, algorithmenübergreifend vergleichbar |
| **Return** | Optimierungsziel des Agenten | **nur innerhalb desselben Reward-Schemas vergleichbar** |

Der Return darf nie zwischen Reward-Schemata verglichen werden — verschiedene
Schemata vergeben per Konstruktion verschieden viele Punkte. Zwischen
Algorithmen ist der Score die gemeinsame Währung, und `steps_to_<n>` die
Aussage über Sample-Effizienz.

## Zwei Frame-Limits, und warum

`max_episode_steps` (Standard 3.000) begrenzt Episoden **im Training**, damit
eine gute Policy nicht das ganze Budget in einer Episode verbraucht.
`eval_max_episode_steps` (Standard 20.000) gilt **beim Messen**.

Das muss getrennt sein. Im `dqn_v2`-Lauf meldete die Evaluation ab 550.000
Schritten immer wieder 79 Röhren bei exakt 3.000,0 Frames mittlerer
Episodenlänge — das war nicht das Können des Agenten, sondern das
Trainingslimit. Mit 20.000 Frames gemessen liegt dieselbe Policy bei 304.

Ein zensierter Score staucht alle guten Varianten auf denselben Wert und macht
jeden Vergleich wertlos. Deshalb meldet `eval.csv` jetzt eine
`truncation_rate`, und `evaluate.py` warnt, sobald Episoden ins Limit laufen.

## Reward-Schemata

Die Reward-Funktion ist keine Konstante im Spielcode mehr, sondern eine
`RewardConfig` aus [`rl/rewards.py`](../rl/rewards.py) — demselben Modul, das
auch die PPO-Arbeit benutzt.

| Preset | Idee |
| --- | --- |
| `legacy` | das Original, Prioritätskette. Baseline des Vergleichs |
| `additive` | dieselben Terme, aber summiert |
| `sparse` | nur Röhre und Tod. Am schwersten zu lernen, aber unverzerrt |
| `survival` | nur Überleben, kein Röhren-Bonus |
| `shaped` | `sparse` plus potential-based Shaping zur Lückenmitte |
| `risk_averse` | Tod kostet −5 statt −1 |
| `energy` | Flattern kostet −0,02 pro Schlag |

## Auswerten und Zuschauen

```bash
python -m flappy_bird_gymnasium.dqn.evaluate --checkpoint runs/dqn_v2/best.pt --episodes 50
python -m flappy_bird_gymnasium.dqn.evaluate --checkpoint runs/dqn_v2/best.pt --render
python -m flappy_bird_gymnasium.dqn.plot runs/dqn_v2 runs/dqn_v3 --out vergleich.png
python -m pytest flappy_bird_gymnasium/tests/test_dqn_agent.py -q
```

## Aufbau

| Modul | Aufgabe |
| --- | --- |
| `config.py` | `DQNConfig` — sämtliche Hyperparameter als Dataclass |
| `replay_buffer.py` | Ringpuffer plus `NStepAccumulator` für n-step Returns |
| `model.py` | `QNetwork` (MLP) und `DuelingQNetwork` (V/A-Zerlegung) |
| `agent.py` | ε-greedy, Bellman-Target, Gradientenschritt, Speichern und Laden |
| `env_utils.py` | `make_env` mit Reward-Preset und den zwei Frame-Limits, Seeding, Thread-Begrenzung |
| `train.py` | Trainingsloop, CSV-Logging, Checkpoints, `steps_to_thresholds` |
| `evaluate.py` | Greedy-Durchläufe, optional mit Fenster |
| `experiments.py` | Studien über mehrere Seeds im Prozess-Pool |
| `summarize.py` | gepaarte Evaluation und Aggregation über Seeds |
| `plot.py` | Sechs-Panel-Lernkurven, auch für den Vergleich mehrerer Läufe |

Die Reward-Definitionen liegen bewusst **nicht** hier, sondern in
[`rl/rewards.py`](../rl/rewards.py), gemeinsam mit der PPO-Arbeit.

## Vier Details, die leicht schiefgehen

**`terminated` ≠ `truncated`.** Nur `terminated` (der Vogel ist abgestürzt)
beendet das Bootstrapping. `truncated` heißt, das Schrittlimit war erreicht —
die Episode endet, aber die Zukunft war nicht wertlos. Würde man das als
terminal speichern, lernte der Agent, dass Erfolg wie Sterben aussieht. Der
n-step-Akkumulator hat dafür zwei getrennte Wege:
`push(…, terminated=True)` leert die Queue als Tod, `flush()` mit erhaltenem
Bootstrap.

**Jede Transition trägt ihren eigenen Diskont.** Am Episodenende werden die
noch wartenden Schritte mit *kürzerem* Horizont geleert. Ein globales
`gamma**n` wäre für die falsch, deshalb speichert der Puffer
`gamma**horizon` pro Transition.

**Huber statt MSE.** Der Reward von +0,1 pro Frame summiert sich bei γ = 0,99
zu großen Returns auf. MSE plus große TD-Fehler ergibt Gradienten, die das Netz
zerlegen; `smooth_l1_loss` und Gradient-Clipping halten das stabil.

**Der letzte Checkpoint ist kein Ergebnis.** Die Greedy-Policy oszilliert:
im `dqn_v2`-Lauf zwischen Score 1,4 und 79,0 innerhalb von 50.000 Schritten.
Deshalb `best.pt` statt `latest.pt`, und deshalb mehrere Seeds.

## Nächste Ausbaustufen

- Prioritized Experience Replay
- LIDAR-Beobachtungen (`use_lidar=True`, 180 Dimensionen) mit Frame-Stacking
- Anbindung an `rl/plotting.py` und `rl/summarize.py`, sobald der PPO-Branch
  gemergt ist — dann liegen DQN- und PPO-Ergebnisse in derselben Auswertung
