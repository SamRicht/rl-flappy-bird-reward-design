# Arbeitsstand DQN — Branch `fb_drl_dqn`

Vollständige Dokumentation der DQN-Arbeit: was vorhanden war, was gebaut wurde,
welche Experimente gelaufen sind, was sie ergeben haben, und welche Läufe
vorbereitet sind. Stand: 15.09.2026, vor dem Start der Ablationsstudie.

Das Dokument ist als Grundlage für die Präsentation gedacht. Abschnitt 11
(*Was sich aus welchem Ergebnis schließen lässt*) formuliert die Hypothesen
**vor** den Läufen — damit später nachvollziehbar ist, was eine Vorhersage war
und was eine nachträgliche Erklärung.

---

## Inhalt

1. [Kurzfassung](#1-kurzfassung)
2. [Ausgangslage](#2-ausgangslage)
3. [Grundsatzentscheidungen](#3-grundsatzentscheidungen)
4. [Was neu gebaut wurde](#4-was-neu-gebaut-wurde)
5. [Was aus dem Team übernommen wurde](#5-was-aus-dem-team-übernommen-wurde)
6. [Abweichungen vom Lehrbuch-DQN](#6-abweichungen-vom-lehrbuch-dqn)
7. [Alle Hyperparameter](#7-alle-hyperparameter)
8. [Durchgeführte Experimente](#8-durchgeführte-experimente)
9. [Eigene Fehler und was daraus folgt](#9-eigene-fehler-und-was-daraus-folgt)
10. [Methodik für die Endauswertung](#10-methodik-für-die-endauswertung)
11. [Was sich aus welchem Ergebnis schließen lässt](#11-was-sich-aus-welchem-ergebnis-schließen-lässt)
12. [Vorbereitete Läufe](#12-vorbereitete-läufe)
13. [Offene Punkte](#13-offene-punkte)
14. [Reproduktion](#14-reproduktion)

---

## 1. Kurzfassung

Aus einem Repository ohne Trainingscode wurde ein DQN-Agent gebaut, der im
Mittel **304 Röhren** passiert. Die Zufallspolicy schafft 0.

| Lauf | Konfiguration | Ø Score (50 Episoden) |
| --- | --- | --- |
| Zufall | — | 0,00 |
| `dqn_v1` | 500k Schritte, Double + Dueling | 7,68 |
| `dqn_v2` | 2M Schritte, Double + Dueling + n-step 3 | **304,26** |

Wichtiger als die Zahl ist die Methodik, die dabei entstanden ist: Studien über
mehrere Seeds, gepaarte Auswertung, zensurfreie Metriken und eine Zufalls-Referenz.
Ohne die sind Vergleiche zwischen Lernverfahren nicht belastbar — was sich an
einem eigenen Fehlschluss gezeigt hat (Abschnitt 9).

**38 Tests**, alle grün.

---

## 2. Ausgangslage

Das Repository ist ein Fork von `markub3327/flappy-bird-gymnasium`.

### Was vorhanden war

- Die fertige Gymnasium-Umgebung `FlappyBird-v0` mit zwei Beobachtungsmodi
- Eine fest verdrahtete Reward-Funktion
- Vortrainierte Keras-Gewichte (`.h5`) und ein Dueling-Netz in `tests/dueling.py`
- `tests/test_dqn.py`, das fertige Gewichte lädt und spielen lässt

### Was nicht vorhanden war

**Kein Trainingscode.** Weder Replay Buffer noch Target-Netzwerk noch
Optimierungsschritt. Das Repository kann einen DQN *ausführen*, aber keinen
*trainieren*.

> **Für die Präsentation relevant:** Der DQN-Trainer ist Eigenarbeit. Übernommen
> wurden die Spielumgebung und — bewusst, siehe Abschnitt 5 — die
> Reward-Definitionen aus der PPO-Arbeit.

### Die Umgebung

| Eigenschaft | Wert |
| --- | --- |
| Beobachtung (genutzt) | 12 normalisierte Features |
| Beobachtung (Alternative) | 180 LIDAR-Strahlen |
| Aktionen | 2 — nichts tun, flattern |
| Bildschirm | 288 × 512 Pixel |
| Röhrengeschwindigkeit | 4 px pro Frame |
| Vogelposition | x = 57, Breite 34 px |

Die 12 Features: horizontale und beide vertikalen Positionen der letzten,
nächsten und übernächsten Röhre (9 Werte), dazu Höhe, Vertikalgeschwindigkeit
und Rotation des Vogels.

---

## 3. Grundsatzentscheidungen

### PyTorch statt TensorFlow

Der vorhandene Modellcode ist Keras, die Gewichte sind `.h5`. Da ohnehin von
Null trainiert wird, fiel die Wahl auf PyTorch: klarerer Trainingsloop,
unkomplizierte Installation unter Python 3.12, und es passt zur PPO-Arbeit im
Team.

### 12 Features statt LIDAR

Die Feature-Variante kommt mit einem kleinen MLP aus und trainiert auf der CPU
in Minuten statt Stunden. LIDAR bleibt als Erweiterung vorgemerkt.

### Eigenes `dqn/`-Paket statt Einbau in `rl/`

Bewusste Entscheidung für Arbeitsgeschwindigkeit. Der Preis ist eine Dopplung
der Infrastruktur, die für die Endauswertung aufgelöst werden sollte
(Abschnitt 13). Geteilt werden von Anfang an die Reward-Definitionen.

---

## 4. Was neu gebaut wurde

Neues Paket `flappy_bird_gymnasium/dqn/`:

| Modul | Zeilen | Aufgabe |
| --- | --- | --- |
| `config.py` | ~90 | `DQNConfig` — alle Hyperparameter als Dataclass, pro Lauf als `config.json` gespeichert |
| `replay_buffer.py` | ~160 | Ringpuffer auf vorallokierten numpy-Arrays plus `NStepAccumulator` |
| `model.py` | ~60 | `QNetwork` (MLP 12→256→256→2) und `DuelingQNetwork` |
| `agent.py` | ~175 | ε-greedy, Bellman-Target, Gradientenschritt, Target-Sync, Speichern/Laden |
| `env_utils.py` | ~85 | `make_env` mit Reward-Preset und zwei Frame-Limits, Seeding, Thread-Begrenzung |
| `train.py` | ~290 | Trainingsloop, CSV-Logging, Checkpoints, `steps_to_thresholds` |
| `evaluate.py` | ~125 | Greedy-Durchläufe, optional mit Fenster; warnt bei zensiertem Score |
| `experiments.py` | ~260 | Studien über mehrere Seeds im Prozess-Pool |
| `summarize.py` | ~290 | gepaarte Auswertung, Aggregation über Seeds, Zufalls-Referenz |
| `plot.py` | ~230 | Lernkurven für Einzelläufe, Seed-Bänder für Studien |
| `tests/test_dqn_agent.py` | ~290 | 38 Tests |

### Der Agent

Standard-DQN mit drei Erweiterungen, jede einzeln abschaltbar:

- **Double DQN** — das Online-Netz wählt die Aktion, das Target-Netz bewertet
  sie. Entfernt die Überschätzung durch den Max-Operator.
- **Dueling-Kopf** — `Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)`. In Flappy Bird
  ist in den meisten Zuständen die Aktion egal; die Trennung von „wie gut ist
  diese Position" und „welche Aktion ist hier besser" lernt schneller.
- **n-step Returns** — statt eines Einzelschritts der Return über n Schritte,
  Bootstrap mit `gamma**n`.

### Der n-step-Akkumulator

Der heikelste Teil, deshalb ausführlich getestet. Er faltet Einzelschritte zu
n-Schritt-Transitionen. Zwei Details, die leicht falsch gemacht werden:

**Zwei getrennte Wege für Episodenenden.** `push(..., terminated=True)` leert
die Warteschlange als Tod, `flush()` leert sie mit erhaltenem Bootstrap. Ein
Abbruch durch das Schrittlimit ist kein Tod — die Zukunft war nicht wertlos.
Würde man das als terminal speichern, lernte der Agent, dass Erfolg wie Sterben
aussieht.

**Jede Transition trägt ihren eigenen Diskont.** Am Episodenende werden die noch
wartenden Schritte mit *kürzerem* Horizont geleert. Ein globales `gamma**n` wäre
für die falsch, deshalb speichert der Puffer `gamma**horizon` pro Transition.
Das ist der Grund, warum n-step- und Einzelschritt-Transitionen im selben Puffer
koexistieren können.

### Die Studien-Infrastruktur

`experiments.py` fährt ein Gitter von Läufen in einem Prozess-Pool, jeder Worker
mit genau einem Torch-Thread — sonst konkurrieren die Läufe um Kerne und jeder
einzelne wird langsamer, als er allein wäre.

Vier Studientypen:

| Studie | Was sie variiert |
| --- | --- |
| `seeds` | nur den Seed — misst die Streuung des Algorithmus selbst |
| `ablation` | Double, Dueling und n-step einzeln und gemeinsam abgeschaltet |
| `sweep` | einen Hyperparameter über eine Werteliste |
| `reward` | die Reward-Presets |

Sweepbar sind: `learning_rate`, `gamma`, `n_step`, `batch_size`, `buffer_size`,
`target_update_interval`, `epsilon_decay_steps`, `learning_starts`,
`train_freq`, `pipe_gap`, `hidden`.

### Die Auswertung

`summarize.py` wertet jeden Lauf einer Studie aus und aggregiert über die Seeds.
Drei Regeln machen den Vergleich belastbar:

1. **Gepaarte Auswertung** — alle Läufe auf denselben Evaluations-Seeds, also
   identischen Röhrenfolgen. Ohne das kann eine Variante gewinnen, weil sie die
   leichteren Level gezogen hat.
2. **Unzensierte Messung** — mit dem hohen Frame-Limit, nicht dem des Trainings.
3. **Zufalls-Referenz** — eine Zeile mit der Zufallspolicy, an der alle Zahlen
   verankert werden.

Ausgabe: `evaluations.csv` (pro Lauf) und `aggregate.csv` (pro Variante), beide
mit einer `algorithm`-Spalte, damit sie später mit den Ergebnissen von PPO,
Q-Learning und CNN zusammengeführt werden können.

---

## 5. Was aus dem Team übernommen wurde

Der Branch `trainingsumgebung_update` wurde gemergt. Damit kommen
`rl/rewards.py` und der zugehörige Umbau der Umgebung dazu.

**Warum übernommen statt nachgebaut:** DQN und PPO trainieren dadurch gegen
exakt dieselben Reward-Definitionen. Zwei eigene Implementierungen derselben
Reward-Funktion wären eine Fehlerquelle, die jeden Vergleich zwischen den
Verfahren wertlos machen könnte.

Die Reward-Funktion ist damit keine Konstante im Spielcode mehr. `step()`
registriert nur noch, *was passiert ist* — Röhre passiert, abgestürzt, Decke
berührt, geflattert — und übergibt diese Fakten an `compute_reward`.

### Die sieben Presets

| Preset | alive | pipe | death | ceiling | Besonderheit |
| --- | --- | --- | --- | --- | --- |
| `legacy` | +0,1 | +1,0 | −1,0 | −0,5 | Prioritätskette, das Original |
| `additive` | +0,1 | +1,0 | −1,0 | −0,5 | Terme werden summiert |
| `sparse` | 0 | +1,0 | −1,0 | 0 | nur die Aufgabe selbst |
| `survival` | +0,1 | 0 | −1,0 | 0 | kein Röhrenbonus |
| `shaped` | 0 | +1,0 | −1,0 | 0 | plus Shaping zur Lückenmitte |
| `risk_averse` | +0,1 | +1,0 | −5,0 | −0,5 | Tod kostet fünffach |
| `energy` | +0,1 | +1,0 | −1,0 | −0,5 | Flattern kostet −0,02 |

`legacy` und `additive` unterscheiden sich nur darin, wie die Terme verknüpft
werden: Bei der Prioritätskette gewinnt der höchstpriorisierte Term allein, bei
`additive` werden alle summiert. Wer im selben Schritt eine Röhre passiert und
stirbt, bekommt unter `legacy` −1,0, unter `additive` 0,0.

---

## 6. Abweichungen vom Lehrbuch-DQN

Fünf Abweichungen, jede aus einem konkreten Problem dieser Umgebung heraus.
Diese Liste eignet sich direkt für die Präsentation.

### 6.1 Huber-Loss statt MSE

Der Reward von +0,1 pro Frame summiert sich bei γ = 0,99 auf: Der Wert des
Ewig-Überlebens ist 0,1 ÷ (1 − 0,99) = **10**. MSE plus große TD-Fehler ergibt
Gradienten, die das Netz zerlegen. `smooth_l1_loss` ist jenseits von 1 linear
statt quadratisch; dazu Gradient-Clipping bei 10,0.

### 6.2 `terminated` strikt getrennt von `truncated`

Nur ein Absturz beendet das Bootstrapping in der Bellman-Gleichung. Das
Schrittlimit beendet die Episode, aber nicht den Wert der Zukunft. Siehe
Abschnitt 4.

### 6.3 Diskont pro Transition statt global

Siehe Abschnitt 4.

### 6.4 Zwei Frame-Limits

`max_episode_steps` (3.000) begrenzt Episoden **im Training**, damit eine gute
Policy nicht das ganze Budget in einer Episode verbraucht.
`eval_max_episode_steps` (20.000) gilt **beim Messen**.

Der Grund ist gemessen, nicht theoretisch: Im `dqn_v2`-Lauf meldete die
Evaluation ab 550.000 Schritten immer wieder 79 Röhren bei exakt 3.000,0 Frames
mittlerer Episodenlänge. Das war nicht das Können des Agenten, sondern das
Trainingslimit. Mit 20.000 Frames gemessen liegt dieselbe Policy bei 304.

Ein zensierter Score staucht alle guten Varianten auf denselben Wert. `eval.csv`
meldet deshalb eine `truncation_rate`, und `evaluate.py` warnt, sobald Episoden
ins Limit laufen.

### 6.5 Der Agent seedet den globalen Torch-Generator selbst

Siehe Abschnitt 9.2.

---

## 7. Alle Hyperparameter

Vollständige Liste mit Begründung. Geänderte Werte sind markiert.

### Netz und Optimierung

| Parameter | Wert | Bedeutung |
| --- | --- | --- |
| `hidden` | (256, 256) | zwei versteckte Schichten, ReLU |
| `dueling` | `True` | V/A-Zerlegung im Kopf |
| `double_dqn` | `True` | Aktionswahl und Bewertung getrennt |
| `learning_rate` | 1e-4 | Adam |
| `batch_size` | 64 | |
| `gamma` | 0,99 | Diskontfaktor |
| `huber_loss` | `True` | siehe 6.1 |
| `grad_clip` | 10,0 | Norm-Clipping |

### Erfahrungsspeicher

| Parameter | Wert | Bedeutung |
| --- | --- | --- |
| `buffer_size` | 200.000 | **geändert** von 100.000 |
| `learning_starts` | 5.000 | Schritte, bevor gelernt wird |
| `train_freq` | 1 | ein Gradientenschritt pro Umgebungsschritt |
| `n_step` | 3 | **strittig**, siehe Abschnitt 8.3 |

### Exploration und Targets

| Parameter | Wert | Bedeutung |
| --- | --- | --- |
| `epsilon_start` | 1,0 | |
| `epsilon_end` | 0,01 | |
| `epsilon_decay_steps` | 200.000 | **geändert** von 150.000 |
| `target_update_interval` | 1.000 | Gradientenschritte zwischen Target-Kopien |

### Umgebung und Messung

| Parameter | Wert | Bedeutung |
| --- | --- | --- |
| `pipe_gap` | 100 | Lückengröße — die Schwierigkeitsachse |
| `reward_preset` | `legacy` | |
| `max_episode_steps` | 3.000 | Trainingslimit |
| `eval_max_episode_steps` | 20.000 | **neu**, Messlimit |
| `eval_episodes` | 20 | **geändert** von 10 |

### Begründung der Änderungen

| Parameter | vorher | nachher | Warum |
| --- | --- | --- | --- |
| `epsilon_decay_steps` | 150.000 | 200.000 | `dqn_v1` räumte die erste Röhre erst bei ~250.000 Schritten zuverlässig. Früher heruntergefahrene Exploration hungert den Agenten an genau den Transitionen aus, aus denen er lernen muss. |
| `buffer_size` | 100.000 | 200.000 | Bei Episoden von 50 Frames fasste der alte Puffer nur ~2.000 Episoden — zu wenig, um seltene erfolgreiche Durchgänge zu halten. |
| `eval_episodes` | 10 | 20 | Mit 10 Episoden konnte ein Glückslauf eine tatsächlich bessere Policy schlagen. Konkret passiert, siehe 9.1. |
| Episodenlimit | Score 100 | 3.000 Frames | Umstellung auf `TimeLimit`, passend zu `rl/envs.py` der PPO-Arbeit. |

---

## 8. Durchgeführte Experimente

### 8.1 `dqn_v1` — erste Baseline

500.000 Schritte, 23 Minuten, Double + Dueling, n-step 1, Reward `legacy`.

| Schritte | Ø Score | Episodenlänge | Return |
| --- | --- | --- | --- |
| 0 – 100k | 0,00 | 50,0 | −4,6 |
| 100k – 200k | 0,07 | 52,0 | +3,7 |
| 200k – 300k | 0,75 | 74,2 | +7,0 |
| 300k – 400k | 2,85 | 151,5 | +16,6 |
| 400k – 500k | 3,09 | 158,5 | +17,5 |

Endmessung über 50 Episoden: **Ø 7,68**, Median 5, Maximum 29, Minimum 1.
Der Endstand-Checkpoint kam auf 6,46.

#### Befund: die erste Röhre ist eine Wand bei Frame 50

Die Episodenlänge klebte in den ersten 100.000 Schritten bei exakt 50,0 Frames.
Das ist Geometrie: Der Vogel steht bei x = 57 und ist 34 px breit, die erste
Röhre startet bei x = 288 und fliegt mit 4 px pro Frame heran.
**(288 − 91) ÷ 4 = 49,25.**

Der untrainierte Agent stirbt also nicht am Boden, sondern exakt an der ersten
Röhre. Die Zufalls-Referenz bestätigt es: Score 0,00 bei Episodenlänge 50,0.

Praktische Folge: Die `+1` für eine Röhre muss bei Einzelschritt-Updates durch
rund 50 Bellman-Backups sickern, bevor sie die erste Aktion einer Episode
beeinflussen kann.

#### Befund: der Agent optimiert zuerst das Billige weg

In den ersten 100.000 Schritten stieg der Return von −7,24 auf +3,15, während
der Score unverändert 0 blieb. Der Agent lernte in dieser Phase nicht, Röhren zu
passieren, sondern **das Flattern zu vermeiden**: Die −0,5-Strafe für die
Bildschirmoberkante ist dicht und sofort spürbar, die +1,0 für eine Röhre ist
selten und liegt 50 Schritte in der Zukunft.

Das ist ein lokales Optimum, das direkt aus der Reward-Konstruktion folgt — und
damit das zentrale Argument für die Reward-Studie.

### 8.2 `dqn_v2` — große Baseline

2.000.000 Schritte, 76 Minuten, Double + Dueling + n-step 3, Reward `legacy`.
Von 11.460 Episoden endeten 6.229 mit mindestens einer passierten Röhre.

| Schritte | Ø Score | bester | Ø Länge | Loss | Flap-Rate |
| --- | --- | --- | --- | --- | --- |
| 0 – 200k | 0,01 | 2 | 50,6 | 0,809 | 0,246 |
| 200k – 400k | 1,37 | 20 | 99,1 | 0,259 | 0,096 |
| 400k – 600k | 4,13 | 41 | 200,7 | 0,219 | 0,096 |
| 600k – 800k | 6,43 | 44 | 286,4 | 0,154 | 0,117 |
| 800k – 1,0M | 7,71 | 46 | 335,6 | 0,122 | 0,123 |
| 1,0M – 1,2M | 7,20 | 46 | 315,1 | 0,107 | 0,140 |
| 1,2M – 1,4M | 6,82 | 54 | 301,2 | 0,102 | 0,140 |
| 1,4M – 1,6M | 7,18 | 74 | 315,0 | 0,102 | 0,132 |
| 1,6M – 1,8M | 6,92 | 69 | 304,9 | 0,096 | 0,144 |
| 1,8M – 2,0M | 7,54 | 56 | 328,2 | 0,103 | 0,149 |

Endmessung über 50 Episoden mit 20.000-Frame-Limit: **Ø 304,26**, Median 292,
Maximum 530, Minimum 0. Der Endstand-Checkpoint kam über 30 Episoden auf 210,33.

#### Der scheinbare Stillstand ab 800k täuscht

Der Trainings-Score verharrt bei rund 7, die Greedy-Evaluation lag zur selben
Zeit bei 30 bis 70. Kein Widerspruch, sondern Arithmetik: Bei ε = 0,01 und 300
Frames Episodenlänge fallen im Schnitt drei Zufallsaktionen an, und eine falsche
Aktion genügt zum Tod.

**Merkposten für die Präsentation:** Trainings-Score und Greedy-Score sind
verschiedene Größen. Wer sie verwechselt, unterschätzt jeden Agenten mit
Restexploration.

#### Die Greedy-Policy oszilliert erheblich

Ab 500.000 Schritten: Median 33,4, Standardabweichung 22,9, Spitzenwert 79,0.
Vier Einbrüche unter 10 — am tiefsten **1,40 bei 1.400.000 Schritten**, direkt
nach 69,4 fünfzigtausend Schritte zuvor.

Das ist bekanntes DQN-Verhalten, kein Aufbaufehler. Folge: Der letzte Checkpoint
eines Laufs ist kein verlässliches Ergebnis, und ein einzelner Lauf pro Variante
beweist nichts.

### 8.3 n-step — zwei widersprüchliche Messungen

#### Erster Versuch: ein Seed je Arm

200.000 Schritte, identische Einstellungen außer `n_step`.

| Schritte | n = 1 | n = 3 |
| --- | --- | --- |
| 100k – 125k | 0,004 | 0,039 |
| 125k – 150k | 0,016 | 0,125 |
| 150k – 175k | 0,039 | 0,421 |
| 175k – 200k | 0,210 | 0,399 |

Episoden mit mindestens einer Röhre bis 200.000 Schritte: 124 gegen 387.

**Dieses Ergebnis wurde zurückgezogen.** Beide Arme hatten genau einen Seed.

#### Replikation: drei Seeds je Wert

300.000 Schritte, gepaarte Auswertung über 30 identische Episoden.

| n-step | Ø Score über Seeds | Spanne | Schritte bis Score 1, je Seed |
| --- | --- | --- | --- |
| **1** | **3,2 ± 1,6** | 1 – 5 | 165k · 167k · 232k |
| 3 | 2,2 ± 2,6 | 0 – 5 | 217k · 239k · 270k |
| 5 | 2,3 ± 1,2 | 1 – 3 | 260k · 262k · 281k |

Beim Endscore liegt alles innerhalb der Seed-Streuung — die
Standardabweichungen (1,2 bis 2,6) sind größer als die Abstände zwischen den
Varianten. Bei „Schritte bis Score 1" ist die Ordnung dagegen konsistent: Der
schlechteste n=1-Seed ist immer noch schneller als der mittlere n=3-Seed, und
kein n=5-Seed erreicht die Marke vor 260.000 Schritten.

**Einschränkungen, die dazugehören:** Keine exakte Replikation — der erste
Versuch lief über 200.000 Schritte mit Puffer 100.000, die Replikation über
300.000 mit Puffer 200.000. Und 300.000 Schritte messen nur die Frühphase;
`dqn_v2` nahm erst nach etwa 250.000 Schritten Fahrt auf.

**Konsequenz:** n-step wird nicht vorab gesetzt, sondern läuft als Variante in
der Ablationsstudie mit. Die Antwort kommt dann mit Fehlerbalken aus dem
Hauptexperiment.

### 8.4 Durchsatz

| Aufbau | pro Lauf | gesamt |
| --- | --- | --- |
| ein Lauf allein | ~480 Schritte/s | ~480 Schritte/s |
| 9 Läufe parallel | ~192 Schritte/s | ~1.730 – 2.100 Schritte/s |

Gemessen auf 14 physischen Kernen. Parallelisierung bringt also gut den
vierfachen Gesamtdurchsatz — für dieselbe Zeit, die der einzelne 2M-Lauf
brauchte, bekommt man neun Läufe über je 300.000 Schritte.

---

## 9. Eigene Fehler und was daraus folgt

Dieser Abschnitt ist bewusst enthalten. Die Fehler sind methodisch lehrreich und
sollten in der Präsentation nicht fehlen — sie begründen, warum die Auswertung so
aufgebaut ist, wie sie aufgebaut ist.

### 9.1 Zu wenige Evaluations-Episoden

Das Trainingslog von `dqn_v1` meldete einen Evaluations-Score von **13,60**.
Über 50 Episoden nachgemessen liefert derselbe Checkpoint **7,68**. Die 13,60
waren Rauschen aus 10 Episoden.

*Folge:* `eval_episodes` auf 20 erhöht; Zahlen für einen Bericht kommen aus
`evaluate.py` oder `summarize.py` mit mindestens 30 Episoden.

### 9.2 Seeds waren nicht reproduzierbar

Ein neu geschriebener Test deckte auf, dass zwei Agenten mit demselben Seed
**unterschiedliche** Aktionen lieferten. Der Agent seedete seinen eigenen
Zufallsgenerator, die Netzgewichte kamen aber aus dem globalen Torch-Generator.
In `train.py` wurde zufällig vorher global geseedet, sodass es dort funktionierte
— aber für eine Seed-Studie ist „Seed 3 reproduziert nicht" fatal.

*Folge:* `DQNAgent.__init__` seedet den globalen Torch-Generator selbst.

### 9.3 Das eigene Frame-Limit verfälschte die Messung

Siehe 6.4. Ein Score von 79 war in Wahrheit ein Score von 304.

*Folge:* getrennte Limits, `truncation_rate` im Log, Warnung in `evaluate.py`.

### 9.4 Aus einem Ein-Seed-Vergleich wurde eine Empfehlung

Siehe 8.3. Die n-step-Empfehlung ging in den 2M-Baseline-Lauf ein, bevor sie
jemals repliziert war.

*Folge:* `experiments.py` und `summarize.py`; keine Aussage mehr ohne mehrere
Seeds.

### 9.5 Drei Fehleinschätzungen während laufender Trainings

Bei 200.000 Schritten von `dqn_v1` lautete die Prognose, 500.000 Schritte würden
nicht reichen — der Agent brach bei 250.000 aus dem Plateau aus. Beim n-step-A/B
sah n-step nach 75.000 Schritten schlechter aus. Bei `dqn_v2` sah der Lauf nach
180.000 Schritten schwächer aus als ein Kontrolllauf — verglichen wurde aber
mitten in der ε-Decay-Phase gegen einen Lauf mit kürzerem ε-Zeitplan.

Alle drei waren zu pessimistisch.

*Folge:* RL-Lernkurven sind über weite Strecken flach und springen dann. Eine
Momentaufnahme sagt fast nichts, und der Vergleich zweier Läufe ist nur bei
identischem Explorationsstand gültig. Urteile gehören ans Ende eines Laufs.

---

## 10. Methodik für die Endauswertung

Vier Lernverfahren sollen verglichen werden: **PPO, DQN, Q-Learning und CNN.**

### 10.1 Welche Zahl wofür

| Metrik | Aussage | Grenze |
| --- | --- | --- |
| **Score** | was die fertige Policy kann | wird vom Frame-Limit zensiert |
| **`steps_to_<n>`** | wie schnell gelernt wurde | nicht zensierbar — gemeinsame Währung über alle vier Verfahren |
| **Return** | das Optimierungsziel des Agenten | **nur innerhalb desselben Reward-Schemas vergleichbar** |

`steps_to_<n>` ist die Zahl der Umgebungsschritte, bis der gleitende Score über
20 Episoden ein Niveau erstmals hielt. Der gleitende Mittelwert verhindert, dass
eine einzelne Glücksepisode als „erreicht" zählt.

**Der Return darf nie zwischen Reward-Schemata verglichen werden** —
verschiedene Schemata vergeben per Konstruktion verschieden viele Punkte.

Die Zufalls-Referenz (Score 0,00) verankert alle Zahlen.

### 10.2 Warum mehrere Seeds

Die Greedy-Policy von `dqn_v2` schwankte zwischen Score 69,4 und 1,4 innerhalb
von 50.000 Schritten. Wer zwei Varianten mit je einem Seed vergleicht,
vergleicht, welcher Seed gezogen wurde. Die Streuung *zwischen* den Seeds ist
der Maßstab, an dem ein Unterschied gemessen wird.

**Faustregel für die Präsentation:** Wenn sich die Fehlerbalken zweier Varianten
über die ganze Länge überlappen, ist kein Unterschied gezeigt — egal, wie weit
die Mittelwerte auseinanderliegen.

### 10.3 Gepaarte Auswertung

Alle Läufe werden auf denselben Evaluations-Seeds gemessen, sehen also identische
Röhrenfolgen. Ohne diese Paarung kann eine Variante gewinnen, weil sie die
leichteren Level gezogen hat.

### 10.4 Was das Team noch entscheiden muss

**Was ist ein faires Budget?** „Gleich viele Umgebungsschritte" heißt bei PPO mit
acht parallelen Umgebungen etwas anderes als bei DQN mit einer. Nach Schritten,
nach Wandzeit oder nach beidem? Diese Entscheidung bestimmt mit, welches
Verfahren gewinnt, und muss *vor* den finalen Läufen stehen.

**Die Beobachtungsräume unterscheiden sich zwangsläufig.** Q-Learning braucht
diskretisierte Zustände, CNN braucht Pixel, DQN und PPO nutzen die 12 Features.
Das ist unvermeidbar — aber `pipe_gap` und das Reward-Preset müssen über alle
vier identisch sein, sonst spielen sie unterschiedliche Spiele.

---

## 11. Was sich aus welchem Ergebnis schließen lässt

Hypothesen, formuliert **vor** den Läufen. Damit ist später unterscheidbar, was
eine Vorhersage war und was eine nachträgliche Erklärung.

### 11.1 Ablationsstudie

Fünf Varianten: `full`, `no_double`, `no_dueling`, `no_nstep`, `vanilla`.

| Wenn … | dann folgt daraus … |
| --- | --- |
| `full` schlägt `vanilla` deutlich | Die Erweiterungen zusammen lohnen sich — die übliche Erwartung aus der Literatur |
| `no_double` fällt ab | Die Überschätzung durch den Max-Operator ist hier relevant. Plausibel, weil der dichte Alive-Reward die Q-Werte groß macht |
| `no_dueling` fällt ab | Die Trennung von Zustandswert und Aktionsvorteil hilft — passend dazu, dass in den meisten Zuständen die Aktion egal ist |
| `no_nstep` ist **nicht** schlechter | Bestätigt die Replikation aus 8.3; n-step bringt in dieser Umgebung nichts. Erklärung: Der Return ist unkorrigiert off-policy, und bei ε = 1,0 am Anfang sind die n-Schritt-Returns stark verrauscht |
| `no_nstep` ist **besser** als `full` | n-step schadet aktiv. Dann gehört n = 1 in die Standardkonfiguration |
| Alle Fehlerbalken überlappen | Bei 1.000.000 Schritten sind die Unterschiede kleiner als das Rauschen. Dann braucht es längere Läufe oder mehr Seeds — kein Ergebnis ist auch ein Ergebnis |

### 11.2 Reward-Studie

| Wenn … | dann folgt daraus … |
| --- | --- |
| `sparse` schlägt `legacy` | Der dichte Alive-Reward schadet mehr, als er nützt — er erzeugt das lokale Optimum aus 8.1. Das wäre das stärkste Ergebnis für die Fragestellung des Repos |
| `legacy` schlägt `sparse` | Der dichte Reward hilft beim Anlernen, trotz des Umwegs über das lokale Optimum |
| `survival` erreicht hohe Scores | Überleben ist ein hinreichendes Ersatzziel — der Agent lernt die Aufgabe, ohne dass man sie ihm nennt. In Flappy Bird erzwingt Überleben Röhrendurchgänge |
| `shaped` lernt schneller als `sparse` | Reines Optimierungsergebnis, da potential-based Shaping die optimale Policy beweisbar nicht verändert. Methodisch der sauberste Befund |
| `shaped` erreicht einen **höheren Endscore** als `sparse` | Widerspruch zur Theorie — dann stimmt etwas an der Shaping-Implementierung oder am Messaufbau nicht. Wäre ein Grund, genauer hinzusehen |
| `energy` senkt die Flap-Rate bei gleichem Score | Der Reward steuert das *Verhalten*, nicht nur das Ergebnis — anschaulichster Einzelbefund für eine Präsentation |
| `risk_averse` erreicht längere Episoden | Der Agent wird vorsichtig und hält sich näher an die Lückenmitte |

### 11.3 Parameter-Sweeps

| Parameter | Erwartung | Was der Befund bedeutet |
| --- | --- | --- |
| `gamma` 0,95 / 0,99 / 0,999 | 0,99 am besten | Bei `legacy` ist der Wert des Ewig-Überlebens 0,1 ÷ (1 − γ), also 2, 10 oder 100. Zu klein → zu kurzsichtig für 50 Frames Vorlauf; zu groß → Q-Werte explodieren und das Lernen wird instabil |
| `target_update_interval` 100 / 1.000 / 5.000 | mittlerer Wert am besten | Zu häufig → das Target jagt sich selbst, zu selten → es ist veraltet. Der Lehrbuch-Stabilitätskompromiss |
| `pipe_gap` 80 / 100 / 130 | monoton fallende Schwierigkeit | Schwierigkeitsachse. **Funktioniert bei allen vier Verfahren** — eignet sich als gemeinsames Gruppenexperiment, aus dem sich ablesen lässt, welches Verfahren mit steigender Schwierigkeit am stabilsten bleibt |
| `epsilon_decay_steps` 100k / 200k / 400k | längerer Decay besser | Es gibt Hinweise aus 8.1, dass zu frühes Herunterfahren den Agenten aushungert |
| `learning_rate` 5e-5 / 1e-4 / 3e-4 | 1e-4 am besten | Klassiker; 3e-4 vermutlich instabil bei den großen Q-Werten |
| `hidden` 64×64 / 256×256 / 512×512 | wenig Unterschied | Bei 12 Eingangsdimensionen ist 256×256 bereits reichlich. Wenn 64×64 gleichauf liegt, ist das ein Argument für kleinere Netze — und damit schnellere Läufe |

---

## 12. Vorbereitete Läufe

### 12.1 Ablationsstudie — startbereit

```powershell
python -m flappy_bird_gymnasium.dqn.experiments ablation `
    --seeds 5 --total-steps 1000000 --eval-interval 50000 --eval-episodes 10
```

25 Läufe (5 Varianten × 5 Seeds), je 1.000.000 Schritte, Reward `legacy`, 12
Prozesse parallel. **Rund vier Stunden.** Platzbedarf etwa 57 MB.

Auswertung danach:

```powershell
python -m flappy_bird_gymnasium.dqn.summarize runs/study_ablation --episodes 30
python -m flappy_bird_gymnasium.dqn.plot runs/study_ablation --study --out ablation.png
```

Das Evaluations-Intervall steht auf 50.000 statt 25.000 und die Episodenzahl auf
10: Bei 25 gleichzeitigen Läufen würde die Zwischenevaluation sonst teurer als
das Training. Die belastbaren Zahlen kommen ohnehin aus `summarize.py`.

### 12.2 Reward-Studie — danach

```powershell
python -m flappy_bird_gymnasium.dqn.experiments reward `
    --rewards legacy sparse shaped energy --seeds 4 --total-steps 1000000
```

16 Läufe, rund zweieinhalb Stunden. Auf vier Presets eingedampft; alle sieben
wären 28 Läufe.

### 12.3 Parameter-Sweep — danach

```powershell
python -m flappy_bird_gymnasium.dqn.experiments sweep --param gamma `
    --values 0.95 0.99 0.999 --seeds 4 --total-steps 1000000
```

12 Läufe, rund zwei Stunden.

---

## 13. Offene Punkte

| Punkt | Status |
| --- | --- |
| Budget-Definition über die vier Verfahren | **offen — Teamentscheidung, vor den Endläufen** |
| Dopplung zwischen `dqn/` und `rl/` | offen; für die gemeinsame Endauswertung auflösen |
| Aufnahmen für die Präsentation | offen; der PPO-Branch hat `rl/record.py`, nach dem Merge übernehmen |
| Prioritized Experience Replay | nicht umgesetzt |
| LIDAR-Beobachtungen mit Frame-Stacking | nicht umgesetzt |

Zur Dopplung: Das `dqn/`-Paket müsste sein Ausgabeformat an das der PPO-Läufe
angleichen (`episodes.csv`, `progress.csv`, `model.pt`), dann arbeiten
`rl/plotting.py` und `rl/summarize.py` auch auf DQN-Läufen. Die Ergebnis-CSVs
tragen bereits eine `algorithm`-Spalte, damit sie sich zusammenführen lassen.

---

## 14. Reproduktion

### Umgebung

Die venv liegt **neben** dem Repo, nicht darin:

```powershell
cd C:\Users\...\Flappy-Bird-KI\rl-flappy-bird-reward-design
..\.venv\Scripts\Activate.ps1
```

Ein blankes `python` benutzt das System-Python, in dem das Paket **nicht**
installiert ist. Neu aufsetzen:

```bash
cd Flappy-Bird-KI
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e rl-flappy-bird-reward-design
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Die gelaufenen Experimente nachstellen

```powershell
# dqn_v1
python -m flappy_bird_gymnasium.dqn.train --run-name dqn_v1 --total-steps 500000 `
    --n-step 1 --epsilon-decay-steps 150000 --buffer-size 100000

# dqn_v2
python -m flappy_bird_gymnasium.dqn.train --run-name dqn_v2 --total-steps 2000000 `
    --n-step 3 --reward legacy --eval-interval 50000 --eval-episodes 15

# n-step-Replikation
python -m flappy_bird_gymnasium.dqn.experiments sweep --param n_step --values 1 3 5 `
    --seeds 3 --total-steps 300000 --epsilon-decay-steps 150000
python -m flappy_bird_gymnasium.dqn.summarize runs/study_sweep --episodes 30
```

### Tests

```powershell
python -m pytest flappy_bird_gymnasium/tests/test_dqn_agent.py -q
```

38 Tests: n-step-Arithmetik, Ringpuffer, Save/Load, alle Reward-Presets,
Reproduzierbarkeit, Frame-Limit-Trennung, Ablationsvarianten, Sweep-Parsing.

### Ausgabe eines Laufs

| Datei | Inhalt |
| --- | --- |
| `config.json` | alle Hyperparameter |
| `train.csv` | pro Episode: Return, Score, Länge, Flap-Rate, ε, Loss, Q-Mittelwert |
| `eval.csv` | periodische Greedy-Auswertung inklusive `truncation_rate` |
| `summary.json` | Kennzahlen inklusive `steps_to_<n>` |
| `best.pt` / `latest.pt` | bester und letzter Checkpoint |

`runs/` steht in `.gitignore` — Checkpoints sind binär und aus Code plus
`config.json` reproduzierbar.
