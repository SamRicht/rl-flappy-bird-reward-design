# Übergabe DQN — Stand 16.09.2026

Arbeitsdokument für die Fortsetzung, nicht für die Präsentation. Die
inhaltlichen Ergebnisse und ihre Deutung stehen in **`ARBEITSSTAND.md`**; dieses
Dokument sagt, wo man steht, was als Nächstes zu tun ist und welche Fallstricke
es gibt.

Branch: `fb_drl_dqn`. Zielgruppe: die nächste Person oder der nächste Agent, der
hier weiterarbeitet.

---

## 1. In einem Absatz

Ein DQN-Agent wurde von Grund auf gebaut (das Repository enthielt keinen
Trainingscode) und in zwei Studien über je 5 Seeds untersucht: welche
Algorithmus-Bausteine etwas bringen (*Ablation*, 25 Läufe) und welches
Belohnungsschema am besten funktioniert (*Reward*, 35 Läufe). Beide sind
ausgewertet und dokumentiert. Als Nächstes folgt die **Parameter-Studie**
(45 Läufe, startbereit), danach ein **finaler Lauf auf neuen Seeds**, dann
Aufnahmen und der Vergleich mit PPO, Q-Learning und CNN.

---

## 2. Sofort loslegen

```powershell
cd C:\Users\1dv95\Flappy-Bird-KI\rl-flappy-bird-reward-design
..\.venv\Scripts\Activate.ps1
python -m pytest flappy_bird_gymnasium/tests/test_dqn_agent.py -q   # 63 Tests
```

Die venv liegt **neben** dem Repo (`Flappy-Bird-KI\.venv`), nicht darin. Ein
blankes `python` nimmt das System-Python, in dem das Paket nicht installiert
ist — das ist die häufigste Stolperfalle.

**Der nächste Lauf** (Details und Begründung in ARBEITSSTAND 12.3):

```powershell
python -m flappy_bird_gymnasium.dqn.experiments params `
    --seeds 5 --total-steps 1000000 --reward shaped `
    --params learning_rate gamma n_step epsilon_decay_steps `
    --eval-interval 50000 --eval-episodes 10 --eval-max-episode-steps 200000 --workers 13
```

45 Läufe, rund 6 bis 6,5 Stunden auf 14 Kernen. Ohne `--params` laufen alle sechs
Parameter des Rasters (65 Läufe, ~8,5 Stunden). Bricht der Lauf ab:
**denselben Befehl erneut starten**, fertige Läufe werden übersprungen.

---

## 3. Was wo liegt

| Pfad | Inhalt |
| --- | --- |
| `flappy_bird_gymnasium/dqn/` | das gesamte DQN-Paket (Eigenarbeit) |
| `flappy_bird_gymnasium/dqn/ARBEITSSTAND.md` | Ergebnisse, Deutung, Hypothesen, Befehle — **die Hauptquelle** |
| `flappy_bird_gymnasium/dqn/HANDOFF.md` | dieses Dokument |
| `flappy_bird_gymnasium/dqn/significance.py` | Permutationstest gegen die Seed-Streuung |
| `flappy_bird_gymnasium/rl/rewards.py` | Reward-Definitionen, **geteilt mit der PPO-Arbeit** — nicht einseitig ändern |
| `flappy_bird_gymnasium/tests/test_dqn_agent.py` | 63 Tests |
| `runs/study_ablation/` | Ablationsstudie, 25 Läufe (ARBEITSSTAND 8.5) |
| `runs/study_reward/` | Reward-Studie, 35 Läufe (ARBEITSSTAND 8.6) |
| `runs/dqn_v1`, `runs/dqn_v2` | frühe Einzelläufe (8.1, 8.2) |
| `runs/study_sweep/` | n-step-Replikation über 300k Schritte (8.3) |

`runs/` steht in `.gitignore`. Die Ergebnisse existieren **nur lokal auf diesem
Rechner** — vor einem Rechnerwechsel sichern, sonst müssen 60 Läufe neu
gerechnet werden.

### Ausgabe je Lauf

`config.json`, `train.csv` (pro Episode), `eval.csv` (periodische Greedy-Messung),
`summary.json` (inklusive `steps_to_<n>`), `best.pt`, `latest.pt`.

### Ausgabe je Studie

`study.json`, `summaries.json`, dazu aus `summarize.py`: `evaluations.csv` und
`aggregate.csv`. Der Dateiname trägt, was die Messung unterscheidet:
`_latest` für den letzten Checkpoint, `_limit200000` für ein erhöhtes Messlimit.
Für die Reward-Studie liegen zusätzlich die Präsentationsgrafiken
`reward_greedy.png` und `reward_speed.png` dort, für die Ablation
`ablation_greedy.png`.

---

## 4. Die sechs Regeln, an denen hier alles hängt

Diese Regeln sind aus Fehlern entstanden (ARBEITSSTAND Abschnitt 9). Wer sie
bricht, produziert Zahlen, die nicht halten.

1. **Nie ein Ergebnis aus einem einzelnen Seed.** Jede Aussage braucht mehrere
   Seeds und die Streuung *zwischen* ihnen als Maßstab. Ein einzelner DQN-Lauf
   schwankte zwischen Score 1,4 und 69,4 innerhalb von 50.000 Schritten.
2. **Einen Seed wählt man nicht aus.** Der beste Seed ist Glück, kein Ergebnis.
3. **Immer `truncation_rate` prüfen.** Ist sie > 0, ist der Score nach oben
   abgeschnitten und muss mit höherem `--max-episode-steps` neu gemessen werden.
4. **Der Return ist nie zwischen Reward-Schemata vergleichbar.** Verglichen
   werden Score, Flap-Rate und `steps_to_<n>`.
5. **Mehrere Messungen, bevor etwas als Befund gilt.** Bester Checkpoint,
   letzter Checkpoint und der Verlauf können sich widersprechen — in der
   Ablation hing der Double-DQN-Befund allein am besten Checkpoint.
6. **Auf neuen Seeds berichten, was auf alten Seeds ausgewählt wurde.** Sonst
   ist die Zahl zu optimistisch (`--seed-offset 100`).

---

## 5. Fallstricke im Code

| Falle | Was passiert |
| --- | --- |
| Läufe mitten im Training fortsetzen | Geht nicht. Checkpoints enthalten weder Replay Buffer noch RNG-Zustand. Abgebrochene Läufe starten neu, fertige werden übersprungen |
| Zwei Studien in dasselbe `--runs-root` | Gleichnamige Varianten überschreiben sich. Je Studie ein eigenes Verzeichnis |
| `--workers` zu hoch | Jeder Lauf belegt einen Kern mit genau einem Torch-Thread. Mehr Worker als Kerne macht alle langsamer. 13 auf 14 Kernen ist der eingespielte Wert |
| Studie zu groß planen | Rechnen mit: 13 Läufe gleichzeitig, ~100 min je Lauf über 1M Schritte, also ~1,7 h je Durchgang. Läufe aufrunden auf volle Durchgänge — der 40. Lauf kostet so viel wie der 52. |
| Laufzeit über weniger Seeds sparen | Mit 3 statt 5 Seeds steigt der kleinstmögliche p-Wert von 0,008 auf 0,1; damit ist nichts mehr nachweisbar. Lieber Varianten streichen |
| Zwischenevaluation als Ergebnis lesen | `eval.csv` misst mit dem **Trainings**-Limit (3.000 Frames ≈ 79 Röhren) und ist oben gedeckelt. Belastbare Zahlen kommen aus `summarize.py` |
| Trainings-Score mit Greedy-Score verwechseln | `train.csv` enthält ε-greedy-Episoden (ε = 0,01). Bei 300 Frames Länge fallen im Schnitt drei Zufallsaktionen an, eine genügt zum Tod. Der Trainings-Score liegt deshalb systematisch unter dem Greedy-Score |
| `rl/rewards.py` ändern | Die Datei ist mit der PPO-Arbeit geteilt (Colins Branch `trainingsumgebung_update`). Eine Änderung macht die Verfahren unvergleichbar — stattdessen `reward_overrides` in der DQN-Konfiguration nutzen |
| `pipe_gap` verändern | 100 ist der Standardwert des Originalspiels. Ein anderer Wert verändert das Spiel und macht Vergleiche mit allen bisherigen Zahlen ungültig |

---

## 6. Entschieden — nicht neu aufrollen

| Entscheidung | Begründung |
| --- | --- |
| PyTorch statt Keras/TensorFlow | von Null trainiert, klarerer Trainingsloop (ARBEITSSTAND 3) |
| 12 Features statt LIDAR | kleines MLP, CPU-tauglich |
| Double + Dueling + n-step 3 als Basis | Ablation, 8.5 |
| Dueling bleibt, obwohl ohne Nutzen | schadet nicht, hält die Konfiguration über alle Studien stabil |
| `shaped` als Basis der Parameter-Studie | schnellstes und bestes Schema, 8.6 |
| 1.000.000 Schritte je Lauf | eingespieltes Budget aller Studien, ~100 min je Lauf |
| 5 Seeds (0 – 4) je Variante | gepaart über alle Studien hinweg |
| `pipe_gap` 100 | Standardwert des Originalspiels |
| Trainingslimit 3.000 Frames | hält Episoden bezahlbar |

---

## 7. Offen — und wer entscheidet

| Punkt | Wer |
| --- | --- |
| Faires Budget für den Vergleich der vier Verfahren (Schritte? Wandzeit?) | **Team**, vor den Endläufen (ARBEITSSTAND 10.4) |
| Gemeinsames Reward-Schema für den Verfahrensvergleich | **Team** (12.8). Für „bester DQN-Agent" ist es `shaped`/`survival`, für den fairen Vergleich vermutlich `legacy` |
| Kontrollierter Nachweis, dass die Deckenstrafe die Ursache ist | optional, Lauf ist startbereit (12.5) |
| Zusammenführung `dqn/` und `rl/` | für die gemeinsame Endauswertung (13) |
| Aufnahmen des Agenten | `rl/record.py` aus dem PPO-Branch übernehmen |

---

## 8. Git

Auf `fb_drl_dqn` liegen **nicht committete Änderungen** (Studientypen `params`
und `reward_terms`, `reward_overrides`, getrennte Ergebnisdateien, Tests,
Dokumentation). Vor dem nächsten großen Lauf committen, damit die Ergebnisse
einem Code-Stand zugeordnet werden können.

Colins Branch `trainingsumgebung_update` wurde nach unserem Merge neu
geschrieben (force-push) und sitzt jetzt auf dem aufgeräumten `main`.
**Inhaltlich sind `rl/rewards.py` und die Umgebung identisch** mit unserem
Stand — geprüft mit `git diff -w`. Beim späteren Merge nach `main` sind aber
Konflikte in README, `setup.py` und den Assets zu erwarten; das sollte im Team
gemacht werden, nicht nebenbei.

---

## 9. Nach dem Parameter-Lauf: was zu tun ist

1. **Auswerten**, beide Checkpoints:
   ```powershell
   python -m flappy_bird_gymnasium.dqn.summarize runs/study_params --episodes 30
   python -m flappy_bird_gymnasium.dqn.summarize runs/study_params --episodes 30 --checkpoint latest.pt
   python -m flappy_bird_gymnasium.dqn.significance runs/study_params --baseline baseline
   python -m flappy_bird_gymnasium.dqn.plot runs/study_params --study --out runs/study_params/params.png
   ```
2. **Gewinner bestimmen.** Eine Variante zählt nur, wenn sie die Basis in
   mehreren Messungen schlägt (Regel 5). Bei 13 Varianten gegen dieselbe Basis
   sind Zufallstreffer zu erwarten — im Zweifel bei der Basis bleiben.
3. **Ergebnisse in ARBEITSSTAND.md** als Abschnitt 8.7 dokumentieren, mit
   Abgleich gegen die Hypothesen aus 11.4. Das Schema der Abschnitte 8.5 und 8.6
   übernehmen: Tabelle, Signifikanz, Deutung, Hypothesen-Abgleich.
4. **Finalen Lauf** starten (12.4), mit `--seed-offset 100` und Messlimit
   500.000.
5. **Aufnahmen** erstellen und den Vergleich mit dem Team vorbereiten.

Der Signifikanztest liegt als `significance.py` im Paket und reproduziert die
p-Werte aus ARBEITSSTAND 8.5 und 8.6:

```powershell
python -m flappy_bird_gymnasium.dqn.significance runs/study_ablation --baseline full
python -m flappy_bird_gymnasium.dqn.significance runs/study_reward --baseline legacy --file evaluations_limit200000.csv
```
