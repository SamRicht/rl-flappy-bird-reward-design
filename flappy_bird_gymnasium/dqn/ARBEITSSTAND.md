# Arbeitsstand DQN — Branch `fb_drl_dqn`

Vollständige Dokumentation der DQN-Arbeit: was vorhanden war, was gebaut wurde,
welche Experimente gelaufen sind, was sie ergeben haben, und welche Läufe
vorbereitet sind. Stand: 16.09.2026, nach Ablations- und Reward-Studie
(Abschnitte 8.5 und 8.6).

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

**Zwei Studien mit je 5 Seeds pro Variante sind ausgewertet:**

*Ablation* (25 Läufe, Abschnitt 8.5): Die drei Erweiterungen zusammen
verneunfachen den Score gegenüber Lehrbuch-DQN (164 gegen 18). Den größten und
stabilsten Einzelbeitrag liefern **n-step Returns** — entgegen der eigenen
Vorhersage. Double DQN hilft beim besten Checkpoint, der Dueling-Kopf bringt
messbar nichts.

*Reward* (35 Läufe, Abschnitt 8.6): Das Reward-Schema wirkt noch stärker als die
Algorithmus-Bausteine — Faktor 6 zwischen bestem und schlechtestem Schema.
`shaped` (442 Röhren) und `survival` (437) schlagen das Original `legacy` (170)
deutlich und lernen rund 40 % schneller. Auffällig: Die drei besten Schemata
sind genau die drei **ohne Deckenstrafe**. Und `survival` erreicht seine 437
Röhren, **ohne dass das Passieren einer Röhre je belohnt wird**.

**63 Tests**, alle grün. Reproduzierbarkeit ist belegt: Die `legacy`-Läufe der
Reward-Studie sind Zeile für Zeile identisch mit den `full`-Läufen der Ablation.

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
| `config.py` | ~155 | `DQNConfig` — alle Hyperparameter als Dataclass, pro Lauf als `config.json` gespeichert |
| `replay_buffer.py` | ~160 | Ringpuffer auf vorallokierten numpy-Arrays plus `NStepAccumulator` |
| `model.py` | ~60 | `QNetwork` (MLP 12→256→256→2) und `DuelingQNetwork` |
| `agent.py` | ~175 | ε-greedy, Bellman-Target, Gradientenschritt, Target-Sync, Speichern/Laden |
| `env_utils.py` | ~150 | `make_env` mit Reward-Schema und zwei Frame-Limits, Seeding, Thread-Begrenzung |
| `train.py` | ~315 | Trainingsloop, CSV-Logging, Checkpoints, `steps_to_thresholds` |
| `evaluate.py` | ~105 | Greedy-Durchläufe, optional mit Fenster; warnt bei zensiertem Score |
| `experiments.py` | ~410 | sechs Studientypen über mehrere Seeds im Prozess-Pool |
| `summarize.py` | ~375 | gepaarte Auswertung, Aggregation über Seeds, Zufalls-Referenz |
| `plot.py` | ~250 | Lernkurven für Einzelläufe, Seed-Bänder für Studien |
| `significance.py` | ~185 | Permutationstest: hält ein Unterschied der Seed-Streuung stand? |
| `tests/test_dqn_agent.py` | ~535 | 63 Tests |

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

Sechs Studientypen:

| Studie | Was sie variiert |
| --- | --- |
| `seeds` | nur den Seed — misst die Streuung des Algorithmus selbst |
| `ablation` | Double, Dueling und n-step einzeln und gemeinsam abgeschaltet |
| `sweep` | einen Hyperparameter über eine Werteliste |
| `params` | mehrere Hyperparameter, jeder für sich um eine gemeinsame Basis herum |
| `reward` | die Reward-Presets |
| `reward_terms` | einzelne Reward-Terme (Überlebensbonus, Deckenstrafe) als 2×2 |

`reward_terms` gibt es, weil sich die Presets in mehreren Termen gleichzeitig
unterscheiden. Ein Unterschied zwischen zwei Presets lässt sich deshalb keinem
einzelnen Term zuschreiben. Über `reward_overrides` in der Konfiguration wird ein
Term eines Presets ersetzt, alles andere bleibt gleich.

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

### 6.6 Der Shaping-Diskont folgt dem gamma des Agenten

Nur beim Preset `shaped` relevant. Potential-based Shaping lässt die optimale
Policy beweisbar unverändert (Ng et al., 1999) — aber nur, wenn sein
Diskontfaktor **derselbe** ist wie der des Lernverfahrens. Die Umgebung kann das
nicht prüfen, deshalb setzt `env_utils.reward_config_for` beide aus einer
Quelle. Beim Standardwert 0,99 stimmte es zufällig ohnehin; in der
Parameter-Studie mit γ = 0,95 und 0,999 (12.3) wäre die Garantie sonst
stillschweigend verletzt.

Mit n-step Returns bleibt das richtig: Die Terme der einzelnen Schritte
teleskopieren zu `gamma**n * Φ(s_n) − Φ(s_0)`, der passende Diskont ist also der
pro Schritt.

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
| `pipe_gap` | 100 | Lückengröße zwischen oberer und unterer Röhre in Pixeln — Standardwert des Originalspiels, unverändert |
| `reward_preset` | `legacy` | Belohnungsschema, siehe Abschnitt 5 |
| `reward_overrides` | `{}` | einzelne Terme des Presets ersetzen, z. B. `{"ceiling": 0.0}` — für die Term-Studie |
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

### 8.5 Ablationsstudie

5 Varianten × 5 Seeds, je 1.000.000 Schritte, Reward `legacy`, sonst die
Konfiguration aus Abschnitt 7. Endmessung mit `summarize.py`: 30 gepaarte
Episoden pro Lauf (Evaluations-Seed 90.000), 20.000-Frame-Limit.
Grafik: `runs/study_ablation/ablation_greedy.png`.

#### Drei Messungen, weil eine allein täuschen kann

| Variante | bester Checkpoint | letzter Checkpoint | Greedy-Verlauf ab 550k |
| --- | --- | --- | --- |
| `full` | **163,8 ± 48,4** (103 – 209) | 71,0 ± 62,0 | **38,1** |
| `no_dueling` | 217,7 ± 176,0 (82 – 517\*) | 78,6 ± 43,9 | 38,5 |
| `no_double` | 64,3 ± 30,0 (18 – 102) | 87,2 ± 102,0 | 30,4 |
| `no_nstep` | 56,4 ± 64,8 (6 – 160) | 48,4 ± 64,1 | 15,2 |
| `vanilla` | 17,9 ± 13,8 (5 – 35) | 12,7 ± 13,4 | 9,7 |
| Zufall | 0,0 | 0,0 | — |

Mittelwert ± Standardabweichung **zwischen den Seeds**, in Klammern schlechtester
bis bester Seed. \* `no_dueling_seed3` lief in 93 % der Episoden ins
20.000-Frame-Limit; sein Wert ist eine Untergrenze. Ohne ihn: 143,0 ± 63,8.

- **Bester Checkpoint** (`best.pt`): was eine Variante im besten Moment kann.
- **Letzter Checkpoint** (`latest.pt`): was man bekommt, wenn man bei 1M einfach
  stoppt. Wegen der Oszillation (8.2) stark verrauscht.
- **Greedy-Verlauf**: Mittel der zehn Zwischenmessungen ab 550.000 Schritten je
  Seed, aus `eval.csv`. Hängt von keiner Checkpoint-Wahl ab, ist aber beim
  Trainingslimit (≈ 79 Röhren) nach oben gedeckelt.

#### Signifikanz: `full` gegen jede Variante

Exakter Permutationstest (Mann-Whitney) über die 5 Seed-Werte je Variante.
Bei n = 5 gegen 5 ist **p = 0,008 der kleinstmögliche Wert** — er bedeutet:
jeder `full`-Seed liegt über jedem Seed der anderen Variante.

| Vergleich | bester Checkpoint | letzter Checkpoint | Greedy-Verlauf |
| --- | --- | --- | --- |
| gegen `vanilla` | **0,008** | **0,032** | **0,008** |
| gegen `no_nstep` | **0,032** | 0,31 | **0,008** |
| gegen `no_double` | **0,008** | 1,0 | 0,22 |
| gegen `no_dueling` | 1,0 | 0,69 | 1,0 |

Vier Vergleiche gegen dieselbe Referenz: Nach Holm-Korrektur hält beim besten
Checkpoint `no_nstep` (0,032) die 5-%-Schwelle nicht mehr. Deshalb zählt ein
Befund hier nur, wenn er in mehreren Messungen auftaucht.

#### Was die Ablationsstudie zeigt

**1. Die Erweiterungen zusammen lohnen sich — eindeutig.** `full` schlägt
`vanilla` in allen drei Messungen, beim besten Checkpoint um den Faktor 9.
Das ist das robusteste Ergebnis der Studie.

**2. n-step ist der wichtigste Einzelbaustein.** Ohne n-step fällt der
Greedy-Verlauf von 38 auf 15. Alle fünf `full`-Seeds liegen über allen fünf
`no_nstep`-Seeds. Nur 3 von 5 `no_nstep`-Seeds halten überhaupt einmal
Score 10 im Training, bei `vanilla` 1 von 5, bei allen anderen 5 von 5. Plausible
Erklärung: Die +1 für eine Röhre liegt ~50 Frames in der Zukunft (8.1).
Mit n = 3 braucht sie ein Drittel so viele Bellman-Backups, bis sie die
entscheidenden Aktionen erreicht.

**3. Double DQN hilft der Spitzenleistung, nicht dem Durchschnitt.** Beim besten
Checkpoint liegt jeder `full`-Seed über jedem `no_double`-Seed (164 gegen 64).
Im Verlauf und am Ende verschwindet der Unterschied. Lesart: Double DQN hebt, wie
gut die Policy in ihren besten Phasen wird, verhindert aber nicht, dass sie
danach wieder einbricht. Das ist ein Hinweis, kein Beleg.

**4. Der Dueling-Kopf bringt nichts Messbares.** In keiner Messung ein
Unterschied (p ≥ 0,69); `no_dueling` liegt sogar leicht vorn. Da das Netz ohne
Dueling-Kopf kleiner ist, spricht in dieser Umgebung nichts für ihn.

**5. Die Erweiterungen machen nicht schneller, sondern besser.** Bis Score 1
brauchen alle Varianten 245k – 310k Schritte (Median), bis Score 5 rund 430k –
470k. Die Unterschiede entstehen erst danach — im Niveau und in der
Stabilität. Bei `vanilla` fallen 21 von 50 späten Zwischenmessungen unter
Score 5, bei `full` eine.

**6. Der Trainings-Score verdeckt das.** In der Trainingskurve
(`ablation.png`) liegen `full`, `no_double` und `no_dueling` gleichauf. Die
Restexploration (ε = 0,01) deckelt alle bei ~8 Röhren (8.2). Unterschiede
zwischen den guten Varianten sieht man nur in Greedy-Messungen.

**7. Das Verhalten ändert sich mit.** Flap-Rate der Greedy-Policy: `full` 0,060,
`no_nstep` 0,080, `vanilla` 0,093, Zufall 0,367. Bessere Varianten flattern
sparsamer.

#### Abgleich mit den Hypothesen aus 11.1

| Hypothese vor den Läufen | Ergebnis |
| --- | --- |
| `full` schlägt `vanilla` deutlich | **bestätigt**, in allen drei Messungen |
| `no_double` fällt ab | **teilweise** — nur beim besten Checkpoint |
| `no_dueling` fällt ab | **nicht bestätigt** — kein Unterschied |
| `no_nstep` ist nicht schlechter | **widerlegt** — n-step ist der größte Einzelbeitrag |
| `no_nstep` ist besser als `full` | **widerlegt** |
| Alle Fehlerbalken überlappen | nein, außer bei Dueling |

#### Konsequenzen

- Standardkonfiguration bleibt Double + n-step 3. Dueling kann für den Vergleich
  mit den anderen Verfahren bleiben (es schadet nicht), ist aber kein Argument.
- Für den Vergleich mit PPO, Q-Learning und CNN nicht nur den besten Checkpoint
  berichten: Der letzte Checkpoint schwankt bei `full` zwischen 17 und 163.
- `no_dueling_seed3` zeigt, dass 20.000 Frames für die besten Policies schon
  wieder zensieren. Für Endzahlen das Limit anheben.

### 8.6 Reward-Studie

Alle 7 Presets × Seeds 0 – 4, je 1.000.000 Schritte, Konfiguration `full`,
`pipe_gap` 100. Endmessung: 30 gepaarte Episoden je Lauf (Evaluations-Seed
90.000). Grafiken: `runs/study_reward/reward_greedy.png` (Verlauf und Endscore)
und `reward_speed.png` (Lerngeschwindigkeit).

#### Ergebnis

| Preset | Endscore, unzensiert | schlechtester – bester Seed | letzter Checkpoint | Verlauf ab 550k | bis Score 10 |
| --- | --- | --- | --- | --- | --- |
| `shaped` | **442 ± 373** | 165 – 983 | 171 | 57,3 | **395k** |
| `survival` | **437 ± 422** | 146 – 1.173 | **290** | 51,2 | 471k |
| `sparse` | 242 ± 130 | 106 – 378 | 222 | **59,2** | 424k |
| `legacy` | 170 ± 55 | 103 – 231 | 73 | 38,1 | 678k |
| `risk_averse` | 142 ± 81 | 54 – 271 | 105 | 39,5 | 596k |
| `additive` | 85 ± 65 | 32 – 197 | 32 | 34,8 | 655k |
| `energy` | 70 ± 52 | 29 – 157 | 63 | 27,1 | 713k |
| Zufall | 0,0 | — | 0,0 | — | nie |

Endscore mit 200.000-Frame-Limit gemessen, damit keine Zensur mehr auftritt
(`aggregate_limit200000.csv`); bei 50.000 Frames liefen `shaped_seed3`,
`survival_seed3` und `sparse_seed1` noch ins Limit. „bis Score 10" ist der
Median über die Seeds.

Bester Einzellauf: **`survival_seed3` mit 1.173 Röhren im Mittel**, im besten
Fall 4.535 — mit einem Reward, der das Passieren von Röhren überhaupt nicht
belohnt.

#### Signifikanz gegen `legacy`

Exakter Permutationstest über die 5 Seeds; 0,008 ist der kleinstmögliche Wert.

| Preset | Endscore | letzter Checkpoint | Verlauf | bis Score 5 | bis Score 10 |
| --- | --- | --- | --- | --- | --- |
| `sparse` | 0,42 | **0,032** | **0,008** | **0,008** | **0,008** |
| `shaped` | 0,42 | 0,095 | **0,008** | **0,008** | **0,008** |
| `survival` | 0,22 | **0,032** | 0,095 | **0,008** | **0,016** |
| `risk_averse` | 0,69 | 0,84 | 0,84 | **0,008** | 0,55 |
| `additive` | 0,056 | 0,42 | 0,84 | 0,42 | 1,0 |
| `energy` | **0,032** (zugunsten `legacy`) | 0,84 | 0,22 | 0,69 | 0,31 |

**Der Endscore allein zeigt fast nichts** — `sparse` liegt mit 242 gegen 170
vorn und erreicht trotzdem nur p = 0,42, weil die Streuung zwischen den Seeds
riesig ist. Belastbar wird der Befund über die zensurfreie
Lerngeschwindigkeit und den Verlauf. Das ist genau der Grund, warum
`steps_to_<n>` mitgemessen wird.

#### Was die Reward-Studie zeigt

**1. Das Reward-Schema wirkt stärker als alles aus der Ablation.** Zwischen
bestem und schlechtestem Schema liegt Faktor 6 (442 gegen 70); die drei
DQN-Erweiterungen zusammen brachten Faktor 9, einzelne Bausteine deutlich
weniger. Wer die Belohnung falsch stellt, verliert mehr als durch einen
fehlenden Algorithmus-Baustein.

**2. Die drei besten Schemata sind genau die drei ohne Deckenstrafe.**
`sparse`, `shaped` und `survival` haben `ceiling = 0`, die vier schlechteren
haben −0,5. Das passt zum Befund aus 8.1: Der Agent lernt zuerst, das Flattern
zu vermeiden, weil die Deckenstrafe dicht und sofort spürbar ist, während die
Röhrenbelohnung selten ist und 50 Frames in der Zukunft liegt.

Der Zusammenhang geht über die bloße Gruppierung hinaus — er ist **monoton in
der Zahl der Strafen aufs Flattern**:

| Preset | Strafen aufs Flattern | bis Score 5 | bis Score 10 |
| --- | --- | --- | --- |
| `shaped` | keine | 320k | **395k** |
| `sparse` | keine | 334k | 424k |
| `survival` | keine | 346k | 471k |
| `risk_averse` | Decke −0,5 | 391k | 596k |
| `additive` | Decke −0,5 | 426k | 655k |
| `legacy` | Decke −0,5 | 435k | 678k |
| `energy` | Decke −0,5 **und** Flattern −0,02 | 496k | **713k** |

Die Trennung ist **vollständig**: Jedes Schema ohne Deckenstrafe lernt schneller
als jedes Schema mit ihr, bei beiden Schwellen. Über alle 35 Läufe gerechnet ist
der Unterschied hochsignifikant (Randomisierungstest, p < 0,001; auf Ebene der
sieben Schemata p = 0,057, weil es nur sieben sind). `energy`, das als einziges
Schema zwei Strafen aufs Flattern legt, ist das langsamste von allen.

Das passt zum Mechanismus aus 8.1: Der Vogel fällt, Flattern ist die einzige
Aktion, die ihn oben hält. Eine Strafe fürs Flattern ist dicht und sofort
spürbar, die Belohnung für eine Röhre ist selten und liegt 50 Frames voraus.
Der Agent optimiert deshalb zuerst das Billige weg — im `dqn_v1`-Lauf stieg der
Return von −7,24 auf +3,15, während der Score bei 0 blieb.

**Was das nicht ist: ein kontrollierter Beweis.** Die Schemata unterscheiden
sich in mehreren Termen gleichzeitig (`sparse` hat auch keinen Überlebensbonus,
`survival` keinen Röhrenbonus), der Befund ist also eine sehr konsistente
Korrelation über sieben Schemata, kein isolierter Nachweis. Ein solcher bräuchte
die Term-Studie aus 12.5, bei der sich benachbarte Varianten in genau einem Term
unterscheiden. Für die Präsentation sollte der Satz deshalb lauten: *„Alle
Schemata ohne Strafe aufs Flattern lernen schneller als alle mit — den
kontrollierten Einzelnachweis haben wir aus Zeitgründen nicht geführt."*

**3. Überleben genügt als Ersatzziel.** `survival` belohnt ausschließlich
Überleben, nie das Passieren einer Röhre — und erreicht 437 Röhren. In Flappy
Bird erzwingt Überleben das Passieren. Anschaulichster Einzelbefund der Studie.

**4. Shaping beschleunigt, verändert die Aufgabe aber nicht.** `shaped` ist das
schnellste Schema (395k bis Score 10 gegen 424k bei `sparse`), der Unterschied
zu `sparse` ist aber nicht signifikant (p = 0,55). Das ist theoriekonform:
Potential-based Shaping lässt die optimale Policy unverändert und kann nur die
Optimierung beschleunigen. Ein höherer Endscore bei endlichem Budget ist deshalb
kein Widerspruch zur Theorie — anders, als in Hypothese 11.2 vermutet.

**5. `energy` ist der einzige klare Verlierer — und steuert das Verhalten
nicht.** Die Flap-Kosten von −0,02 senken den Score (p = 0,032 zugunsten
`legacy`), aber die Flap-Rate bleibt unverändert: alle Schemata liegen zwischen
0,060 und 0,066 (Zufall: 0,367). Die Flugphysik gibt die Flatterrate praktisch
vor; ein Preis pro Flügelschlag macht das Lernen nur schwerer.

**6. Die Verknüpfung der Terme ist fast egal.** `additive` unterscheidet sich von
`legacy` im Verlauf nicht (p = 0,84). Der Unterschied betrifft nur den seltenen
Fall „Röhre passiert und im selben Frame gestorben".

**7. Episodenlänge trägt keine eigene Information.** Sie ist durch die Geometrie
an den Score gekoppelt: rund 37,7 Frames je Röhre. Die Hypothese „`risk_averse`
erreicht längere Episoden" ist damit nicht von „erreicht einen höheren Score"
trennbar — und beides trifft nicht zu.

#### Abgleich mit den Hypothesen aus 11.2

| Hypothese vor den Läufen | Ergebnis |
| --- | --- |
| `sparse` schlägt `legacy` | **bestätigt** — bei Lerngeschwindigkeit und Verlauf mit p = 0,008 |
| `legacy` schlägt `sparse` | widerlegt |
| `survival` erreicht hohe Scores | **bestätigt** — 437 Röhren ohne Röhrenbelohnung |
| `shaped` lernt schneller als `sparse` | **tendenziell** — schnellstes Schema, aber nicht signifikant |
| `shaped` mit höherem Endscore ⇒ Widerspruch zur Theorie | **Hypothese war falsch gestellt.** Höherer Endscore bei endlichem Budget folgt aus schnellerem Lernen, nicht aus einer veränderten optimalen Policy |
| `energy` senkt die Flap-Rate bei gleichem Score | **widerlegt** — Flap-Rate unverändert, Score niedriger |
| `risk_averse` erreicht längere Episoden | **widerlegt** — kein Unterschied zu `legacy` |
| `additive` weicht von `legacy` ab | **widerlegt**, wie erwartet |

#### Reproduzierbarkeit

Die fünf `legacy`-Läufe haben dieselbe Trainingskonfiguration wie die
`full`-Läufe der Ablationsstudie. Ihre `train.csv` sind **Zeile für Zeile
identisch** — über 8.000 Episoden je Lauf, in einer getrennten Studie, Tage
später gestartet. Damit ist belegt, dass jede Zahl dieses Dokuments aus Code und
`config.json` reproduzierbar ist.

### 8.7 Durchsatz

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

### 9.5 Die Zufalls-Referenz war geschätzt statt gemessen

Die Referenzzeile trug eine fest eingetragene Flap-Rate von 0,5 — mit dem
Argument, eine Gleichverteilung über zwei Aktionen flattere eben in der Hälfte
der Fälle. Gemessen sind es **0,375**: Der Vogel kann oberhalb des
Bildschirmrands nicht flattern, die Aktion verpufft also manchmal.

*Folge:* Alle Messungen laufen jetzt durch dieselbe `rollout`-Funktion — die
trainierten Agenten und die Referenz. Damit ist die Referenz per Konstruktion
so gemessen wie das, wofür sie die Referenz ist, statt durch sorgfältiges
Abschreiben.

### 9.6 Drei Fehleinschätzungen während laufender Trainings

Bei 200.000 Schritten von `dqn_v1` lautete die Prognose, 500.000 Schritte würden
nicht reichen — der Agent brach bei 250.000 aus dem Plateau aus. Beim n-step-A/B
sah n-step nach 75.000 Schritten schlechter aus. Bei `dqn_v2` sah der Lauf nach
180.000 Schritten schwächer aus als ein Kontrolllauf — verglichen wurde aber
mitten in der ε-Decay-Phase gegen einen Lauf mit kürzerem ε-Zeitplan.

Alle drei waren zu pessimistisch.

*Folge:* RL-Lernkurven sind über weite Strecken flach und springen dann. Eine
Momentaufnahme sagt fast nichts, und der Vergleich zweier Läufe ist nur bei
identischem Explorationsstand gültig. Urteile gehören ans Ende eines Laufs.

### 9.7 Die Replikation maß die falsche Phase

Die n-step-Replikation (8.3) lief über 300.000 Schritte und legte nahe, n-step
bringe nichts. Die Hypothese in 11.1 übernahm das. In der Ablationsstudie über
1.000.000 Schritte ist n-step der größte Einzelbeitrag (8.5). Die Replikation war
nicht falsch gemessen, sie maß nur die Frühphase. Dort unterscheiden sich die
Varianten kaum; die Unterschiede entstehen erst ab rund 500.000 Schritten.

Dass die Hypothese vorab aufgeschrieben war, macht diesen Fehler sichtbar,
statt ihn nachträglich wegzuerklären.

*Folge:* Kurze Pilotläufe taugen, um Fehler zu finden, nicht, um Varianten
auszusortieren. Vergleiche laufen über das volle Budget.

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
| `additive` weicht von `legacy` ab | Schon die Verknüpfung der Terme (Prioritätskette gegen Summe) ändert das Lernen. Erwartung: kein Unterschied, weil die beiden nur im seltenen Fall „Röhre passiert und im selben Frame gestorben" auseinanderliegen |
| `survival` erreicht hohe Scores | Überleben ist ein hinreichendes Ersatzziel — der Agent lernt die Aufgabe, ohne dass man sie ihm nennt. In Flappy Bird erzwingt Überleben Röhrendurchgänge |
| `shaped` lernt schneller als `sparse` | Reines Optimierungsergebnis, da potential-based Shaping die optimale Policy beweisbar nicht verändert. Methodisch der sauberste Befund |
| `shaped` erreicht einen **höheren Endscore** als `sparse` | Widerspruch zur Theorie — dann stimmt etwas an der Shaping-Implementierung oder am Messaufbau nicht. Wäre ein Grund, genauer hinzusehen |
| `energy` senkt die Flap-Rate bei gleichem Score | Der Reward steuert das *Verhalten*, nicht nur das Ergebnis — anschaulichster Einzelbefund für eine Präsentation |
| `risk_averse` erreicht längere Episoden | Der Agent wird vorsichtig und hält sich näher an die Lückenmitte |

### 11.3 Term-Studie (`reward_terms`), formuliert nach 8.6

Die Reward-Studie legt nahe, dass die **Deckenstrafe** die Bremse ist: Die drei
besten Schemata sind genau die ohne sie. Beweisen kann sie das nicht, weil sich
die Presets in mehreren Termen zugleich unterscheiden. Das 2×2 trennt beide
Terme, alles andere bleibt gleich (Modus `additive`, Röhrenbonus und Todesstrafe
immer an).

| Wenn … | dann folgt daraus … |
| --- | --- |
| `no_ceiling` ≈ `neither` und beide schlagen `both` | **Die Deckenstrafe ist die Ursache.** Sie bestraft das Flattern dicht und sofort und drängt den Agenten in das lokale Optimum aus 8.1 |
| `no_alive` ≈ `neither` und beide schlagen `both` | Der Überlebensbonus ist die Ursache: Er macht Q-Werte groß und das Passieren einer Röhre relativ unwichtig |
| Nur `neither` ist gut, die Einzelabschaltungen nicht | Beide Terme zusammen erzeugen das Problem; einer allein genügt nicht |
| Alle vier gleichauf | Der Unterschied zwischen den Presets kommt aus etwas anderem — dann bleibt als Kandidat der Modus (Prioritätskette) oder eine Wechselwirkung mit `gamma` |

### 11.4 Parameter-Studie (Aufbau in 12.3)

Basis ist `shaped` mit der Konfiguration aus Abschnitt 7.

| Parameter | Erwartung | Was der Befund bedeutet |
| --- | --- | --- |
| `gamma` 0,95 / **0,99** / 0,999 | 0,99 am besten | γ legt fest, wie weit der Agent blickt. Zu klein → zu kurzsichtig für die 50 Frames Vorlauf bis zur ersten Röhre; zu groß → die Q-Werte werden groß und das Lernen instabil. Bei `shaped` zieht γ zusätzlich den Shaping-Term mit (6.6) |
| `learning_rate` 5e-5 / **1e-4** / 3e-4 | 1e-4 am besten | Klassiker; 3e-4 vermutlich instabil, 5e-5 langsamer, aber womöglich stabiler — bei einer Policy, die stark oszilliert (8.2), wäre das ein interessanter Befund |
| `n_step` 1 / **3** / 5 | 3 oder 5 | Unter `legacy` war n = 1 klar schlechter (8.5). Unter `shaped` ist das Signal dichter, n-step könnte also weniger wichtig sein. Wenn n = 5 gewinnt, ist die Belohnungsverzögerung noch immer der Engpass |
| `target_update_interval` 250 / **1.000** / 4.000 | mittlerer Wert am besten | Zu häufig → das Target jagt sich selbst, zu selten → es ist veraltet. Der Lehrbuch-Stabilitätskompromiss |
| `epsilon_decay_steps` 100k / **200k** / 400k | offen | Aus 8.1 gibt es Hinweise, dass zu frühes Herunterfahren aushungert. `shaped` lernt aber schneller, deshalb könnte jetzt weniger Exploration genügen — ein Ergebnis zugunsten 100k wäre ein Beleg, dass die Explorationsdauer zum Reward passen muss |
| `hidden` 64×64 / **256×256** / 512×512 | wenig Unterschied | Bei 12 Eingangsdimensionen ist 256×256 reichlich. Wenn 64×64 mithält, ist das ein Argument für kleinere Netze — und schnellere Läufe für alle folgenden Experimente |
| Alle Varianten gleichauf | die Standardwerte sind gut genug | Kein Ergebnis ist auch ein Ergebnis: Dann ist die Konfiguration aus Abschnitt 7 für diese Umgebung bereits nahe am Optimum, und der Reward bleibt der entscheidende Hebel |

**Nicht getestet wird `pipe_gap`.** Die Lückengröße ist der Standardwert des
Originalspiels; sie zu verändern hieße, das Spiel zu verändern. Als gemeinsames
Gruppenexperiment über alle vier Verfahren bleibt sie aber interessant
(Abschnitt 13).

---

## 12. Vorbereitete Läufe

### 12.1 Ablationsstudie — gelaufen, Ergebnisse in 8.5

```powershell
python -m flappy_bird_gymnasium.dqn.experiments ablation `
    --seeds 5 --total-steps 1000000 --eval-interval 50000 --eval-episodes 10 --workers 13
```

25 Läufe (5 Varianten × 5 Seeds), je 1.000.000 Schritte, Reward `legacy`.
Platzbedarf etwa 57 MB.

Auswertung danach:

```powershell
python -m flappy_bird_gymnasium.dqn.summarize runs/study_ablation --episodes 30
python -m flappy_bird_gymnasium.dqn.plot runs/study_ablation --study --out ablation.png
```

Das Evaluations-Intervall steht auf 50.000 statt 25.000 und die Episodenzahl auf
10: Bei 25 gleichzeitigen Läufen würde die Zwischenevaluation sonst teurer als
das Training. Die belastbaren Zahlen kommen ohnehin aus `summarize.py`.

### 12.2 Reward-Studie — gelaufen, Ergebnisse in 8.6

```powershell
python -m flappy_bird_gymnasium.dqn.experiments reward `
    --seeds 5 --total-steps 1000000 --eval-interval 50000 --eval-episodes 10 `
    --eval-max-episode-steps 50000 --workers 13
```

**35 Läufe** (alle 7 Presets × Seeds 0 – 4), je 1.000.000 Schritte, Konfiguration
`full` aus der Ablation. Etwa **5,5 – 6 Stunden** bei 13 Prozessen, rund 80 MB.

**Warum so und nicht anders:**

| Entscheidung | Begründung |
| --- | --- |
| alle 7 Presets statt 4 | Die Reward-Frage ist die Kernfrage des Repos. Mit 13 Prozessen kosten 35 Läufe kaum mehr Wandzeit als 28 (jeweils drei Durchgänge) |
| 5 Seeds, dieselben wie in der Ablation | Die Ablation hat gezeigt, dass die Streuung zwischen Seeds groß ist; jedes Preset sieht dieselben Seeds, der Vergleich ist gepaart |
| Konfiguration `full` eingefroren | Die Reward-Frage wird mit einer festen, getesteten Konfiguration beantwortet. Parameter werden erst danach angefasst (Abschnitt 12.3) |
| `pipe_gap` 100 | Standardwert des Originalspiels — das Spiel bleibt unverändert |
| Messlimit 50.000 statt 20.000 Frames | In der Ablation lief `no_dueling_seed3` in 93 % der Episoden ins 20.000er-Limit. Das Trainingslimit (3.000) ist unverändert, das Training also identisch |
| `legacy` wird neu trainiert | Dieselbe Trainingskonfiguration wie `full` in der Ablation. `legacy_seedN` muss dieselbe `train.csv` liefern wie `full_seedN` — ein kostenloser Reproduzierbarkeitstest |

**So ausgewertet** (die erzeugten Dateien liegen in `runs/study_reward/`):

```powershell
python -m flappy_bird_gymnasium.dqn.summarize runs/study_reward --episodes 30
python -m flappy_bird_gymnasium.dqn.summarize runs/study_reward --episodes 30 --checkpoint latest.pt
python -m flappy_bird_gymnasium.dqn.summarize runs/study_reward --episodes 30 --max-episode-steps 200000
python -m flappy_bird_gymnasium.dqn.significance runs/study_reward --baseline legacy --file evaluations_limit200000.csv
python -m flappy_bird_gymnasium.dqn.plot runs/study_reward --study --out runs/study_reward/reward.png
python -m flappy_bird_gymnasium.dqn.plot runs/study_reward --study --column flap_rate --out runs/study_reward/reward_flap_rate.png
```

| Datei | Inhalt |
| --- | --- |
| `evaluations.csv` / `aggregate.csv` | bester Checkpoint, Limit 50.000 |
| `evaluations_latest.csv` / `aggregate_latest.csv` | letzter Checkpoint |
| `evaluations_limit200000.csv` / `aggregate_limit200000.csv` | **unzensiert — die Zahlen aus 8.6** |
| `reward_greedy.png` | Lernverlauf und Endscore je Seed — die Hauptgrafik |
| `reward_speed.png` | Lerngeschwindigkeit je Seed |
| `reward.png`, `reward_flap_rate.png`, `reward_length.png` | Trainingskurven mit Seed-Bändern |

**Nur Score, Flap-Rate und `steps_to_<n>` vergleichen — nie den Return.** Jedes
Preset vergibt per Konstruktion andere Punktzahlen. Die Episodenlänge ist über
die Geometrie an den Score gekoppelt (≈ 37,7 Frames je Röhre) und trägt keine
eigene Information.

### 12.3 Parameter-Studie — startbereit, der nächste Lauf

Sucht die beste Einstellung für den finalen Lauf. Hyperparameter, jeder für sich
um eine gemeinsame Basis herum variiert.

```powershell
python -m flappy_bird_gymnasium.dqn.experiments params `
    --seeds 5 --total-steps 1000000 --reward shaped `
    --params learning_rate gamma n_step epsilon_decay_steps `
    --eval-interval 50000 --eval-episodes 10 --eval-max-episode-steps 200000 --workers 13
```

**45 Läufe** (9 Varianten × 5 Seeds), rund **6 bis 6,5 Stunden**, etwa 105 MB.

Die Zeit ergibt sich aus der Belegung: 13 Läufe laufen gleichzeitig, jeder
braucht rund 100 Minuten, ein Durchgang dauert also etwa 1,7 Stunden. 45 Läufe
sind drei volle Durchgänge plus einen mit nur 6 Läufen — der letzte ist
schneller, weil sich weniger Läufe die Kerne teilen. Jeder weitere Parameter
kostet 10 Läufe, also rund 1,5 Stunden.

| Parameter | Basis | getestet | im Lauf | Erwartung |
| --- | --- | --- | --- | --- |
| `learning_rate` | 1e-4 | 5e-5, 3e-4 | ja | 1e-4 am besten; 3e-4 vermutlich instabil |
| `gamma` | 0,99 | 0,95, 0,999 | ja | 0,99 am besten; 0,95 zu kurzsichtig für 50 Frames Vorlauf, 0,999 instabil |
| `n_step` | 3 | 1, 5 | ja | 3 oder 5; n = 1 war unter `legacy` klar schlechter (8.5) |
| `epsilon_decay_steps` | 200.000 | 100.000, 400.000 | ja | offen — `shaped` lernt schneller, vielleicht genügt weniger Exploration |
| `target_update_interval` | 1.000 | 250, 4.000 | nein | mittlerer Wert am besten |
| `hidden` | 256×256 | 64×64, 512×512 | nein | wenig Unterschied; wenn 64×64 mithält, ist das ein Argument für kleinere Netze |

Alle sechs Parameter sind im Raster hinterlegt. Ohne `--params` laufen sie alle
(65 Läufe, rund 8,5 Stunden). Die vier ausgewählten sind die mit der größten
erwarteten Wirkung: Lernrate und gamma sind die klassischen Stellschrauben,
n-step war in der Ablation der stärkste Einzelbaustein, und die Explorationsdauer
passt möglicherweise nicht mehr zum schnelleren Reward. Die beiden übrigen lassen
sich einzeln nachziehen, etwa `--params hidden` (15 Läufe, gut 2 Stunden).

**Zwei Abkürzungen, die bewusst *nicht* genommen wurden:**

- **Weniger Seeds.** Mit 3 statt 5 Seeds wären alle sechs Parameter in gut 5 Stunden
  machbar — aber der kleinstmögliche p-Wert stiege von 0,008 auf 0,1. Damit
  wäre kein Unterschied mehr nachweisbar, und die Studie hätte ihren Zweck
  verloren. Lieber weniger Parameter als weniger Seeds.
- **Kürzere Läufe.** Mit 700.000 statt 1.000.000 Schritten passten alle sechs
  Parameter in die Zeit. Genau dieser Fehler ist in 9.7 dokumentiert: Die
  n-step-Replikation über 300.000 Schritte maß die falsche Phase und führte zu
  einer widerlegten Vorhersage. In der Reward-Studie trennten sich die Schemata
  erst ab etwa 400.000 Schritten.

**Warum dieser Aufbau:**

- **Ein Parameter je Variante** („one factor at a time"). Dadurch ist jeder
  Unterschied genau einer Ursache zuzuordnen. Der Preis: Wechselwirkungen —
  zwei Änderungen, die sich nur gemeinsam lohnen — findet der Aufbau nicht.
  Ein volles Gitter würde statt 13 Varianten 3⁶ = 729 kosten.
- **Die Basis läuft nur einmal** und dient allen Parametern als Vergleich. Nur
  deshalb sind sechs Parameter in einem Lauf bezahlbar.
- **Werte links und rechts der Basis.** Nur so lässt sich „der Standardwert ist
  ein Optimum" von „wir haben nie weiter geschaut" unterscheiden.
- **Reward `shaped`**, weil es in 8.6 am schnellsten lernt und den höchsten
  Endscore erreicht. Der Shaping-Diskont folgt `gamma` automatisch (6.6), der
  gamma-Arm bleibt damit theoretisch sauber.
- **5 Seeds wie in allen Studien**, damit die Streuung zwischen den Seeds der
  gewohnte Maßstab bleibt.

**Auswertung danach:**

```powershell
python -m flappy_bird_gymnasium.dqn.summarize runs/study_params --episodes 30
python -m flappy_bird_gymnasium.dqn.summarize runs/study_params --episodes 30 --checkpoint latest.pt
python -m flappy_bird_gymnasium.dqn.significance runs/study_params --baseline baseline
python -m flappy_bird_gymnasium.dqn.plot runs/study_params --study --out runs/study_params/params.png
```

**Wie das Ergebnis zu lesen ist:** Eine Variante gilt nur dann als besser als die
Basis, wenn sie es in mehreren Messungen ist — Endscore, letzter Checkpoint und
`steps_to_<n>`. In der Ablation hing der Double-DQN-Befund allein am besten
Checkpoint und verschwand beim letzten (8.5). Bei acht Varianten gegen dieselbe
Basis ist außerdem mit Zufallstreffern zu rechnen: Bei 5 gegen 5 Seeds liegt der
kleinstmögliche p-Wert bei 0,008, aber wer achtmal testet, findet auch ohne
echten Effekt gelegentlich etwas. Deshalb zählt nur, was deutlich und in
mehreren Messungen auftritt.

### 12.4 Finaler Lauf — danach

Die gewählte Konfiguration auf **neuen Seeds**:

```powershell
python -m flappy_bird_gymnasium.dqn.experiments seeds `
    --seeds 5 --seed-offset 100 --total-steps 1000000 --reward shaped `
    --eval-interval 50000 --eval-episodes 10 --eval-max-episode-steps 500000 `
    --runs-root runs/final --workers 5
    # plus die in 12.3 gefundenen Parameter, z. B. --gamma 0.999
python -m flappy_bird_gymnasium.dqn.summarize runs/final --episodes 50 --max-episode-steps 500000
```

**Warum neue Seeds:** Wer auf den Seeds 0 – 4 die Parameter aussucht und dann auf
denselben Seeds das Ergebnis berichtet, berichtet zu optimistisch — die Wahl hat
sich an den Zufall genau dieser Seeds angepasst. Der finale Lauf misst die
gewählte Konfiguration deshalb auf Seeds, die bei der Auswahl keine Rolle
gespielt haben. Das ist dieselbe Trennung wie zwischen Trainings- und Testdaten.

**Warum Messlimit 500.000:** siehe 12.7.

### 12.5 Term-Studie — gebaut, zurückgestellt

Beantwortet die offene Frage aus 8.6 kontrolliert: Liegt es an der Deckenstrafe?
Zurückgestellt, weil die vorhandenen Daten dafür bereits sehr konsistent sind
(8.6) und die Zeit in die Parametersuche geht. Der Lauf ist startbereit, falls
noch Rechenzeit übrig ist — er ist unabhängig von allem anderen.

```powershell
python -m flappy_bird_gymnasium.dqn.experiments reward_terms `
    --seeds 5 --total-steps 1000000 --eval-interval 50000 --eval-episodes 10 `
    --eval-max-episode-steps 200000 --workers 13
```

**20 Läufe** (4 Varianten × 5 Seeds), rund **3 Stunden**, etwa 45 MB.

| Variante | Überlebensbonus | Deckenstrafe | entspricht |
| --- | --- | --- | --- |
| `both` | +0,1 | −0,5 | Preset `additive` |
| `no_ceiling` | +0,1 | 0 | — |
| `no_alive` | 0 | −0,5 | — |
| `neither` | 0 | 0 | Preset `sparse` |

Alles im Modus `additive`, Röhrenbonus +1,0 und Todesstrafe −1,0 überall gleich.
Damit unterscheiden sich benachbarte Varianten in **genau einem Term** —
anders als die Presets, die sich in mehreren gleichzeitig unterscheiden.
Hypothesen dazu stehen in 11.3.

### 12.6 Fahrplan

| Schritt | Inhalt | Status |
| --- | --- | --- |
| 1 | Grundkonfiguration festlegen (Ablation) | erledigt, 8.5 |
| 2 | Reward-Studie | erledigt, 8.6 |
| 3 | **Parameter-Studie** | **startbereit, 12.3** |
| 4 | Finaler Lauf auf neuen Seeds, unzensiert ausgewertet | danach, 12.4 |
| 5 | Aufnahmen des finalen Agenten | danach; `rl/record.py` aus dem PPO-Branch übernehmen |
| 6 | Zusammenführung mit PPO, Q-Learning, CNN | braucht die Budget-Entscheidung aus 10.4 |
| — | Term-Studie (Ursache des Reward-Effekts) | optional, 12.5 |

### 12.7 Welches Frame-Limit

Das **Trainingslimit bleibt bei 3.000** — es hält Episoden bezahlbar und ist
Teil jeder bisherigen Konfiguration. Verändert wird nur das **Messlimit**.

| Studie | Messlimit | Zensur |
| --- | --- | --- |
| Ablation | 20.000 | ein Lauf zu 93 % zensiert |
| Reward | 50.000 | drei Läufe teilweise zensiert |
| Reward, Nachmessung | 200.000 | keine |
| Parameter-Studie | 200.000 | erwartet: keine |
| **Finaler Lauf** | **500.000** | Reserve, siehe unten |

**Lohnt eine weitere Erhöhung?** Für die Parameter-Studie nein: Bei 200.000
Frames lief in der Reward-Studie keine einzige Episode mehr ins Limit. Die
Reserve ist aber knapper, als die Null vermuten lässt — die beste Episode von
`survival_seed3` dauerte rund 171.000 Frames, also **85 % des Limits**. Ein
Agent, der nach der Parametersuche noch besser wird, überschreitet das.

Für den finalen Lauf deshalb 500.000. Die Kosten sind gering, weil nur die
Episoden teuer sind, die tatsächlich so lange überleben. **Die Prüfung bleibt
immer dieselbe: `truncation_rate` muss 0 sein.** Ist sie größer, ist der Score
nach oben abgeschnitten und muss mit höherem Limit neu gemessen werden.

### 12.8 Welches Reward-Schema wofür

Zwei verschiedene Fragen, zwei verschiedene Antworten:

- **Für den Vergleich der vier Lernverfahren** muss das Schema über alle
  identisch sein — das ist eine Teamentscheidung (10.4). `legacy` ist der
  naheliegende gemeinsame Nenner, weil es das Original reproduziert und die
  PPO-Arbeit es ebenfalls als Baseline führt.
- **Für „was kann unser bester DQN-Agent?"** ist `shaped` oder `survival` die
  Antwort: 442 bzw. 437 Röhren gegenüber 170 bei `legacy`. Der beste Einzellauf
  (`survival_seed3`) kommt auf 1.173.

---

## 13. Offene Punkte

| Punkt | Status |
| --- | --- |
| Budget-Definition über die vier Verfahren | **offen — Teamentscheidung, vor den Endläufen** |
| Woher der Reward-Effekt kommt (Deckenstrafe?) | Daten sprechen dafür (8.6), kontrollierter Nachweis offen; Term-Studie startbereit, 12.5 |
| Gemeinsames Reward-Schema für den Verfahrensvergleich | offen — Teamentscheidung, siehe 12.6 |
| Dopplung zwischen `dqn/` und `rl/` | offen; für die gemeinsame Endauswertung auflösen |
| Aufnahmen für die Präsentation | offen; der PPO-Branch hat `rl/record.py`, nach dem Merge übernehmen |
| Schwierigkeit (`pipe_gap` 80 / 100 / 130) | offen; eignet sich als gemeinsames Experiment über alle vier Verfahren |
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

### Studien erneut starten

`experiments.py` überspringt Läufe, die bereits eine `summary.json` und eine
identische `config.json` haben; alle anderen beginnen von vorn. Damit lässt sich
eine Studie mit demselben Befehl ergänzen. Mitten in einem Lauf fortzusetzen ist
bewusst nicht vorgesehen: Checkpoints enthalten weder Replay Buffer noch
RNG-Zustand, ein fortgesetzter Lauf wäre nicht mit einem durchgelaufenen
vergleichbar.

### Tests

```powershell
python -m pytest flappy_bird_gymnasium/tests/test_dqn_agent.py -q
```

63 Tests: n-step-Arithmetik, Ringpuffer, Save/Load, alle sieben Reward-Presets,
Shaping-Diskont, Reward-Term-Overrides, Aufbau von Reward-, Term- und
Parameter-Studie (ein Parameter je Variante), Überspringen fertiger Läufe,
getrennte Dateien je Messung, Permutationstest,
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
