# Übergabe DQN — Stand 17.09.2026

Arbeitsdokument für die Fortsetzung, nicht für die Präsentation. Die
inhaltlichen Ergebnisse und ihre Deutung stehen in **`ARBEITSSTAND.md`**; dieses
Dokument sagt, wo man steht, was als Nächstes zu tun ist und welche Fallstricke
es gibt.

Branch: `fb_drl_dqn`. Zielgruppe: die nächste Person oder der nächste Agent, der
hier weiterarbeitet.

---

## 1. In einem Absatz

Ein DQN-Agent wurde von Grund auf gebaut (das Repository enthielt keinen
Trainingscode) und in drei Studien über je 5 Seeds untersucht — zusammen mit dem
finalen Lauf **140 Läufe**: welche Algorithmus-Bausteine etwas bringen
(*Ablation*, 25 Läufe), welches Belohnungsschema am besten funktioniert
(*Reward*, 35 Läufe), welche Hyperparameter (*Parameter*, 65 Läufe) und ob die
gewählte Konfiguration auf **neuen Seeds** hält (*final*, 15 Läufe). **Alles ist
gerechnet und ausgewertet.**

Das wichtigste Einzelergebnis für die Übergabe: **Der finale Lauf hat das
Hyperparameter-Tuning widerlegt** (ARBEITSSTAND 8.9). Die ungetunte
Konfiguration aus Abschnitt 7 gewinnt auf den gehaltenen Seeds. Sie ist die
Konfiguration des finalen Agenten — nichts ist mehr zu ändern.

Als Nächstes folgen **Aufnahmen** und der **Vergleich mit PPO, Q-Learning und
CNN**. Rechenzeit wird dafür nicht mehr gebraucht.

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

**Es steht kein Lauf mehr an.** Alle Ergebnisse liegen vor. Um sie
nachzuvollziehen, genügen diese Befehle — sie rechnen nichts neu und laufen in
Sekunden:

```powershell
python -m flappy_bird_gymnasium.dqn.significance runs/final_dqn --baseline baseline
python -m flappy_bird_gymnasium.dqn.significance runs/final_dqn --baseline baseline --file evaluations_latest.csv
python -m flappy_bird_gymnasium.dqn.significance runs/study_ablation --baseline full
python -m flappy_bird_gymnasium.dqn.significance runs/study_reward --baseline legacy --file evaluations_limit200000.csv
```

Der zweite Befehl ist der wichtigste des Projekts: Er zeigt, dass das
Hyperparameter-Tuning auf neuen Seeds **nicht** hält (ARBEITSSTAND 8.9).

**Die Zahl für die Präsentation** — ungetunte Konfiguration, Seeds 100 – 104,
zensurfrei, letzter Checkpoint:

> **Ø 488,7 ± 408,7 Röhren**, Median 504,5, Spanne 48,3 – 1.121,1.
> Zufallsreferenz: 0,00.

**Der Agent für die Aufnahmen** ist ein anderer und darf es sein (Regel 2 gilt
für berichtete Zahlen, nicht für Illustrationen):
`runs/final_dqn/hidden_512x512_seed102/best.pt` — Ø 3.447,7 Röhren, Median
1.823,5, beste Episode 13.273. Dieser Wert ist zensiert und damit eine
Untergrenze.

---

## 3. Was wo liegt

| Pfad | Inhalt |
| --- | --- |
| `flappy_bird_gymnasium/dqn/` | das gesamte DQN-Paket (Eigenarbeit) |
| `flappy_bird_gymnasium/dqn/ARBEITSSTAND.md` | Ergebnisse, Deutung, Hypothesen, Befehle — **die Hauptquelle** |
| `flappy_bird_gymnasium/dqn/HANDOFF.md` | dieses Dokument |
| `flappy_bird_gymnasium/dqn/significance.py` | Permutationstest gegen die Seed-Streuung |
| `flappy_bird_gymnasium/dqn/figures.py` | die fünf Abbildungen für Präsentation und Vergleich |
| `flappy_bird_gymnasium/dqn/record.py` | Greedy-Episode als GIF aufzeichnen |
| `docs/` | erzeugte Abbildungen und Aufnahme — nicht versioniert, jederzeit neu erzeugbar |
| `flappy_bird_gymnasium/rl/rewards.py` | Reward-Definitionen, **geteilt mit der PPO-Arbeit** — nicht einseitig ändern |
| `flappy_bird_gymnasium/tests/test_dqn_agent.py` | 63 Tests |
| `runs/study_ablation/` | Ablationsstudie, 25 Läufe (ARBEITSSTAND 8.5) |
| `runs/study_reward/` | Reward-Studie, 35 Läufe (ARBEITSSTAND 8.6) |
| `runs/study_params/` | Parameter-Studie, 65 Läufe (ARBEITSSTAND 8.7) |
| `runs/final_dqn/` | **finaler Lauf, 15 Läufe auf neuen Seeds (ARBEITSSTAND 8.9)** |
| `runs/dqn_v1`, `runs/dqn_v2` | frühe Einzelläufe (8.1, 8.2) |
| `runs/study_sweep/` | n-step-Replikation über 300k Schritte (8.3) |

`runs/` steht in `.gitignore`. Die Ergebnisse existieren **nur lokal auf diesem
Rechner**: rund 420 MB, 140 Läufe, etwa 24 Stunden Rechenzeit — und ein Stromausfall
hat während der Studien schon einmal zugeschlagen (die Läufe waren danach
reproduzierbar, aber nur, weil sie fertig waren).

**Das ist derzeit das größte Einzelrisiko der Arbeit.** Ein Plattenfehler kostet
die gesamte Auswertungsgrundlage der Präsentation. Von den 371 MB sind 316 MB
Checkpoints und nur 52 MB Text (CSV, JSON, Grafiken). Die Checkpoints sind aus
Code und `config.json` reproduzierbar, die CSVs nicht ohne 20 Stunden Rechnen —
ein Backup der Textdateien genügt also und ist in Sekunden erledigt:

```powershell
robocopy runs <Zielordner>\runs /S /XF *.pt
```

**Eine Ausnahme:** Die `best.pt` aus `runs/final_dqn` wird für die Aufnahmen
gebraucht. `runs/final_dqn` gehört deshalb **vollständig** ins Backup, inklusive
Checkpoints — das sind rund 50 MB:

```powershell
robocopy runs\final_dqn <Zielordner>\runs\final_dqn /S
```

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

## 4. Die sieben Regeln, an denen hier alles hängt

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
   ist die Zahl zu optimistisch (`--seed-offset 100`). **Diese Regel hat sich im
   finalen Lauf als die wichtigste von allen erwiesen:** Zwei Parameter, die in
   8.7 in mehreren Messungen *und* monoton über drei Werte gewonnen hatten,
   kehren sich auf neuen Seeds um (8.9). Weder Konsistenz über Messungen noch
   Monotonie haben die Überanpassung verhindert — nur der gehaltene Seed-Satz.
7. **Ein Messlimit ist eine Annahme über das Können des Agenten.** Wird der
   Agent besser, muss das Limit mitwachsen. Das gilt auch für die
   Zwischenevaluation `quick_eval`, die am Trainingslimit misst (~79 Röhren) —
   ab der Reward-Studie deckelt sie den Greedy-Verlauf und die Auswahl von
   `best.pt` (ARBEITSSTAND 9.9).

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
| `minutes` als Eigenschaft einer Variante lesen | Die Spalte misst, **wann** ein Lauf an der Reihe war. Läufe aus dem letzten, dünn belegten Durchgang wirken doppelt so schnell. Nur innerhalb eines Durchgangs vergleichbar (ARBEITSSTAND 9.8) |
| Messlimit aus der Vorstudie übernehmen | Wird der Agent besser, zensiert das alte Limit wieder. In der Parameter-Studie liefen acht Messungen ins 200.000er-Limit, obwohl die Reward-Studie dort noch zensurfrei war |
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
| `hidden` **256×256** — 512×512 verworfen | Der Stabilitätsgewinn aus 8.7 hielt auf neuen Seeds nicht (8.9). Übrig bleibt nur 18 % schnelleres Lernen zum doppelten Rechenpreis (149,5 gegen 75,8 min) |
| `target_update_interval` **1.000** — 250 verworfen | dasselbe; auf neuen Seeds in keiner Messung mehr ein Effekt (8.9) |
| `epsilon_decay_steps` und `learning_rate` **nicht** geändert | beide nur schneller, nicht besser — bei festem Budget kein Gewinn (8.7) |
| Kein weiteres Hyperparameter-Tuning | 65 Läufe haben am Endscore nichts bewegt, und die zwei scheinbaren Gewinner sind widerlegt. Der Hebel liegt im Reward (8.6), nicht in den Parametern |
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
| Ob Double DQN, Dueling und Huber-Loss unter `shaped` noch etwas bringen | offen; unter `legacy` ausgewählt, nie nachgeprüft (ARBEITSSTAND 8.7, „Eine Altlast"). Ein Satz für die Präsentation |
| Welche Konfiguration in den Verfahrensvergleich geht | **entschieden** — die aus ARBEITSSTAND 7, für jedes Schema. Es gibt keine getunte Variante mehr (8.9). Die Bausteinwahl bleibt schemaabhängig (12.8) |
| `quick_eval` misst am Trainingslimit | offen; deckelt Greedy-Verlauf und `best.pt`-Auswahl (ARBEITSSTAND 9.9). Für den Verfahrensvergleich relevant, falls dort Zwischenmessungen verglichen werden |
| Zusammenführung `dqn/` und `rl/` | für die gemeinsame Endauswertung (13) |
| Aufnahmen des Agenten | `rl/record.py` aus dem PPO-Branch übernehmen |

---

## 8. Git

Der Code ist auf `fb_drl_dqn` committet; `f249d4f` ist der Stand, unter dem alle
140 Läufe entstanden sind — der finale Lauf eingeschlossen. Offen sind nur
Änderungen an diesen beiden Markdown-Dateien.

**Am DQN-Code ist nichts mehr zu ändern**, solange die Zahlen aus 8.5 bis 8.9
gelten sollen (`agent.py`, `train.py`, `env_utils.py`, `model.py`,
`rl/rewards.py`). Die eine bekannte Verbesserung — ein eigenes Limit für
`quick_eval` (ARBEITSSTAND 9.9) — würde alle bisherigen Läufe unvergleichbar
machen und lohnt sich nur, wenn danach ohnehin neu gerechnet wird.

Colins Branch `trainingsumgebung_update` wurde nach unserem Merge neu
geschrieben (force-push) und sitzt jetzt auf dem aufgeräumten `main`.
**Inhaltlich sind `rl/rewards.py` und die Umgebung identisch** mit unserem
Stand — geprüft mit `git diff -w`. Beim späteren Merge nach `main` sind aber
Konflikte in README, `setup.py` und den Assets zu erwarten; das sollte im Team
gemacht werden, nicht nebenbei.

---

## 9. Was jetzt noch zu tun ist

Der finale Lauf ist gerechnet und in ARBEITSSTAND 8.9 ausgewertet. Es fehlt kein
Experiment mehr. Offen sind nur noch Darstellung und Teamarbeit:

1. **Präsentation bauen.** Abbildungen und Aufnahme liegen fertig in `docs/`
   (ARBEITSSTAND 12.4a) und sind mit zwei Befehlen neu erzeugbar:

   ```powershell
   python -m flappy_bird_gymnasium.dqn.figures
   python -m flappy_bird_gymnasium.dqn.record `
       runs/final_dqn/hidden_512x512_seed102/best.pt --seconds 30 --every 2 --scale 0.75
   ```

   Die belastbaren Achsen sind Reward-Schema (Faktor 6), Algorithmus-Bausteine
   (Faktor 9) und Lerngeschwindigkeit. Der Endscore des besten Checkpoints trägt
   wenig — der Grund steht in ARBEITSSTAND 9.9. Der stärkste methodische Punkt
   ist 8.9: ein Befund, der auf gehaltenen Seeds gefallen ist, mit der
   Vorhersage schriftlich vor dem Lauf (12.4).
2. **Vergleich mit dem Team** vorbereiten — dafür fehlt die Budget-Entscheidung
   (ARBEITSSTAND 10.4) und das gemeinsame Reward-Schema (12.8).
   `dqn_lernkurve.png` zeigt das Format, in dem die vier Verfahren
   nebeneinandergelegt werden können; die `algorithm`-Spalte in den
   Ergebnis-CSVs ist dafür schon vorgesehen.

Optional, falls Rechenzeit übrig ist: die Term-Studie (ARBEITSSTAND 12.5, 20
Läufe, ~3 h). Sie führt den kontrollierten Nachweis für den Deckenstrafen-Befund
aus 8.6 — der ist derzeit eine sehr konsistente Korrelation, kein Beweis.

Der Signifikanztest liegt als `significance.py` im Paket und reproduziert die
p-Werte aus ARBEITSSTAND 8.5, 8.6 und 8.9 — die Befehle stehen in Abschnitt 2.
