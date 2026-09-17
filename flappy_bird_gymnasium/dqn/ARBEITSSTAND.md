# Arbeitsstand DQN — Branch `fb_drl_dqn`

Vollständige Dokumentation der DQN-Arbeit: was vorhanden war, was gebaut wurde,
welche Experimente gelaufen sind, was sie ergeben haben, und welche Läufe
vorbereitet sind. Stand: 17.09.2026, **nach dem finalen Lauf** (Abschnitte 8.5
bis 8.9). Alle 140 Läufe sind gerechnet und ausgewertet; als Nächstes folgen
Aufnahmen und der Vergleich mit PPO, Q-Learning und CNN.

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

Aus einem Repository ohne Trainingscode wurde ein DQN-Agent gebaut. Die
Zufallspolicy passiert 0 Röhren; der beste gemessene Einzellauf 1.173.

| Lauf | Konfiguration | Ø Score |
| --- | --- | --- |
| Zufall | — | 0,00 |
| `dqn_v1` | 500k Schritte, Double + Dueling, `legacy` | 7,68 |
| `dqn_v2` | 2M Schritte, + n-step 3, `legacy` | 304,26 |
| `survival_seed3` | 1M Schritte, Reward `survival` (8.6) | **1.173** |

**Die Endzahl der Präsentation** kommt aus dem finalen Lauf (8.9) und ist ein
Mittelwert über fünf Seeds mit Streuung — nicht eine der Zahlen oben, denn
`dqn_v2` ist ein einzelner Lauf unter dem alten Belohnungsschema und
`survival_seed3` der beste aus 35 (Regel: Abschnitt 9, 10.2):

> **Ø 488,7 ± 408,7 Röhren**, Median 504,5, Spanne 48,3 – 1.121,1, gemessen auf
> fünf Seeds, die an keiner Auswahl beteiligt waren, zensurfrei. Zufall: 0,00.

Das ist die Zahl der **ungetunten** Konfiguration aus Abschnitt 7. Die in der
Parameter-Studie gewählte getunte Konfiguration erreichte dort beim letzten
Checkpoint einen Median von 558 bis 662 Röhren gegenüber 186 bei der Basis (8.7)
— **dieser Befund hat den Test auf neuen Seeds nicht überstanden und kehrt sich
dort um** (8.9). Berichtet wird deshalb die einfache Konfiguration; sie ist
zugleich die schnellste.

Wichtiger als die Zahl ist die Methodik, die dabei entstanden ist: Studien über
mehrere Seeds, gepaarte Auswertung, zensurfreie Metriken und eine Zufalls-Referenz.
Ohne die sind Vergleiche zwischen Lernverfahren nicht belastbar — was sich an
einem eigenen Fehlschluss gezeigt hat (Abschnitt 9).

**Drei Studien mit je 5 Seeds pro Variante plus der finale Lauf — 140 Läufe:**

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

*Parameter* (65 Läufe, Abschnitt 8.7): Am **Endscore ändert kein einziger
Hyperparameter etwas** — die Seed-Streuung ist größer als jeder Effekt. Zwei
Parameter verbessern dafür die **Stabilität** deutlich und monoton über alle drei
getesteten Werte: häufigere Target-Kopien (`target_update_interval` 250) und ein
größeres Netz (512×512). Beim letzten Checkpoint — dem, den man bekommt, wenn man
bei 1M einfach stoppt — bringt das 558 bzw. 662 Röhren statt 186. Die Parameter
heben also nicht die Decke, sondern den Boden.

*Finaler Lauf* (15 Läufe, Abschnitt 8.9): Die beiden Parameter-Gewinner werden
auf neuen Seeds **widerlegt** — die ungetunte Basis gewinnt beim letzten
Checkpoint mit Median 504 gegen 289. Der Einbruch, den das Tuning verhindern
sollte, tritt auf den neuen Seeds gar nicht erst auf; er war eine Eigenschaft der
Auswahl-Seeds. Übrig bleibt ein einziger replizierter Effekt: Das große Netz
lernt 18 % schneller — zum doppelten Rechenpreis. **Der gehaltene Seed-Satz hat
damit einen Fehlbefund abgefangen, bevor er in die Präsentation ging.**

Zwei Querbezüge, die erst durch die dritte Studie sichtbar wurden: Unter `shaped`
ist **n-step fast wirkungslos**, obwohl es unter `legacy` der größte
Einzelbaustein war (8.5) — Shaping und n-step lösen dasselbe Problem. Und die
**Explorationsdauer gehört zum Reward**: Unter `shaped` lässt sie sich halbieren,
ohne etwas zu verlieren.

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
| `figures.py` | ~330 | die fünf verfahrensbezogenen Abbildungen für Präsentation und Vergleich |
| `record.py` | ~150 | zeichnet eine Greedy-Episode als GIF auf |
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

**Diese Rechnung gilt für `legacy`**, unter dem die Entscheidung fiel. Die
Schemata `shaped` und `sparse` haben `alive = 0,0`, ihre Q-Werte bleiben also
klein und das Argument trägt dort nicht. Huber-Loss bleibt trotzdem — es ist die
konservative Wahl und schadet bei kleinen Fehlern nicht, weil es dort ohnehin
quadratisch ist. Siehe 8.7, „Eine Altlast, die diese Studie sichtbar macht".

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

**Diese Tabelle gilt für alle Studien — auch nach dem finalen Lauf.** Der finale
Lauf hat zwei Abweichungen geprüft (`hidden` 512×512 und
`target_update_interval` 250, aus 8.7) und **beide verworfen**: Auf neuen Seeds
gewinnt die Konfiguration dieser Tabelle (8.9). Sie ist damit nicht nur die, unter
der 8.5 bis 8.7 entstanden sind, sondern auch die des finalen Agenten.

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

### 8.7 Parameter-Studie

Alle sechs Parameter des Rasters, jeder für sich um eine gemeinsame Basis herum
variiert: **13 Varianten × Seeds 0 – 4 = 65 Läufe**, je 1.000.000 Schritte,
Reward `shaped`, sonst die Konfiguration aus Abschnitt 7. Endmessung mit
`summarize.py`: 30 gepaarte Episoden je Lauf (Evaluations-Seed 90.000),
Messlimit 200.000 Frames.

Grafiken: `runs/study_params/params.png` (Trainingskurven mit Seed-Bändern, aus
`plot.py --study`) und **`params_greedy.png`** — die Hauptgrafik: drei Messungen
nebeneinander, ein Punkt je Seed, bei gleicher Ordnung der Varianten. Sie zeigt
den Kernbefund direkt: Im linken Feld (bester Checkpoint) liegen alle Varianten
übereinander, im mittleren (letzter Checkpoint) trennen sich zwei ab.

Gelaufen in drei Aufrufen (45 Läufe, dann `--params hidden`, dann der Rest des
Rasters). Die Konfigurationen sind davon unberührt — jeder Lauf hat seine eigene
`config.json`, die Basis ist für alle dieselbe, und `study.json` trägt nach dem
letzten Aufruf wieder alle 65 Einträge.

#### Ergebnis

Sortiert nach dem **letzten Checkpoint**, weil nur der hier überhaupt trennt.
Ø ± Standardabweichung zwischen den Seeds, daneben der Median — die
Seed-Verteilungen sind stark schief, einzelne Glücks-Seeds ziehen die
Mittelwerte weit nach oben.

| Variante | bester CP Ø | Median | letzter CP Ø | Median | schlecht. Seed (letzter) | Verlauf ab 550k | bis 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hidden_512x512` | 425,7 ± 249,9 | 374,0 | **783,2 ± 532,9**\* | **558,1** | **279,3** | 57,7 | **331k** |
| `target_update_interval_250` | **927,6 ± 716,9**\* | **981,4** | **604,6 ± 299,6** | **661,7** | **206,1** | **60,9** | 383k |
| `epsilon_decay_steps_400000` | 1.111,8 ± 1.160,0\* | 612,9 | 480,8 ± 901,2\* | 94,5 | 23,1 | 55,0 | 518k |
| `learning_rate_0p0003` | 285,2 ± 76,6 | 275,9 | 414,7 ± 517,5 | 252,4 | 17,9 | **62,5** | **304k** |
| `n_step_1` | 677,4 ± 755,4\* | 518,3 | 335,4 ± 514,1 | 70,5 | 13,8 | 53,1 | 425k |
| `gamma_0p999` | 203,3 ± 187,3 | 141,5 | 242,9 ± 130,9 | 185,3 | 141,5 | 48,1 | 381k |
| `n_step_5` | 204,7 ± 88,9 | 156,7 | 243,9 ± 159,7 | 219,6 | 46,4 | 57,9 | 439k |
| `epsilon_decay_steps_100000` | 363,4 ± 183,6 | 440,5 | 233,1 ± 191,4 | 200,7 | 56,5 | 56,8 | 313k |
| **`baseline`** | **442,2 ± 372,5** | **192,2** | **173,2 ± 96,3** | **186,2** | **30,4** | **57,3** | **395k** |
| `gamma_0p95` | 356,9 ± 507,4 | 143,3 | 169,4 ± 73,9 | 165,2 | 77,3 | 60,4 | 432k |
| `learning_rate_5em05` | 183,6 ± 54,8 | 184,5 | 155,3 ± 96,6 | 162,3 | 63,5 | 44,1 | 508k |
| `target_update_interval_4000` | 464,1 ± 489,3\* | 190,2 | 96,4 ± 56,3 | 74,8 | 49,7 | 58,5 | 400k |
| `hidden_64x64` | 101,3 ± 101,9 | 58,2 | 84,7 ± 100,0 | 58,2 | 0,0 | 28,8 | 606k |
| Zufall | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | — | nie |

\* mindestens ein Seed lief ins 200.000-Frame-Limit; dieser Wert ist eine
**Untergrenze** (Einzelheiten weiter unten). „bis 10" ist der Median der
Schritte bis zum gleitenden Trainings-Score 10, „schlecht. Seed" der schwächste
der fünf Seeds beim letzten Checkpoint — die Zahl, die zählt, wenn man wissen
will, wie schlecht ein Lauf ausfallen kann.

#### Signifikanz gegen `baseline`

Exakter Permutationstest über die 5 Seeds; 0,008 ist der kleinstmögliche Wert.

| Variante | bester CP | letzter CP | bis 5 | bis 10 |
| --- | --- | --- | --- | --- |
| `hidden_512x512` | 0,84 | **0,008** | 0,056 | **0,008** (schneller) |
| `target_update_interval_250` | 0,22 | **0,032** | 0,42 | 0,31 |
| `epsilon_decay_steps_100000` | 1,0 | 0,69 | **0,032** (schneller) | **0,008** (schneller) |
| `epsilon_decay_steps_400000` | 1,0 | 0,55 | **0,008** (langsamer) | **0,008** (langsamer) |
| `learning_rate_0p0003` | 0,69 | 0,42 | **0,032** (schneller) | **0,016** (schneller) |
| `learning_rate_5em05` | 0,31 | 0,84 | **0,032** (langsamer) | **0,008** (langsamer) |
| `n_step_1` | 0,84 | 0,55 | 0,69 | **0,032** (langsamer) |
| `n_step_5` | 0,22 | 0,55 | 0,22 | 0,095 |
| `gamma_0p95` | 0,15 | 0,69 | 0,55 | 0,15 |
| `gamma_0p999` | 0,095 | 0,84 | 0,69 | 0,42 |
| `target_update_interval_4000` | 1,0 | 0,31 | 0,84 | 0,84 |
| `hidden_64x64` | 0,056 | 0,31 | — | — |

**Zur Einordnung der p-Werte:** Bei 12 Varianten gegen dieselbe Basis müsste eine
strenge Holm-Korrektur den kleinsten p-Wert unter 0,05 ÷ 12 = 0,004 drücken. Das
ist bei 5 gegen 5 Seeds **konstruktionsbedingt unerreichbar** — 0,008 ist die
Untergrenze. Kein Befund dieser Studie kann also allein über einen p-Wert
getragen werden. Was ihn trägt, ist Übereinstimmung über mehrere Messungen und
Monotonie über die drei Werte eines Parameters.

#### Was die Parameter-Studie zeigt

**1. Kein Parameter hebt die Spitzenleistung.** Beim besten Checkpoint ist
**keine** Variante von der Basis zu trennen (kleinster p-Wert 0,056, und der
gehört dem Verlierer `hidden_64x64`). Der Grund steht in den Seed-Spalten: Die
Basis erreicht 165 · 187 · 192 · 684 · 983 — zwei Glücks-Seeds tragen den
Mittelwert von 442, der Median liegt bei 192. Bei solchen Verteilungen kann ein
Test über 5 Seeds nichts nachweisen. **Der Endscore des besten Checkpoints ist
in dieser Studie die schwächste aller Messungen**, genau umgekehrt zur Intuition.

**2. Zwei Parameter verbessern die Stabilität — und nur die.** `hidden`
512×512 und `target_update_interval` 250 gewinnen beim **letzten** Checkpoint,
beide mit klarem Abstand und beide auch im Median und im schlechtesten Seed:

> ⚠️ **Dieser Befund ist widerlegt.** Der finale Lauf auf den Seeds 100 – 104 hat
> ihn nicht bestätigt, sondern umgekehrt: Dort gewinnt die ungetunte Basis mit
> Median 504,5 gegen 289,2, und der schlechteste Seed steht bei 48,3 gegen 13,6
> (8.9). Der Abschnitt bleibt unverändert stehen, weil er korrekt berichtet, was
> auf den Seeds 0 – 4 gemessen wurde — und weil der Vergleich mit 8.9 den
> eigentlichen Lehrsatz trägt: **Monotonie über drei Werte schützt nicht vor
> Überanpassung an den Auswahl-Seed-Satz.** Alles Folgende dieses Befundes ist
> unter diesem Vorbehalt zu lesen.

| | letzter CP, Median | schlechtester Seed |
| --- | --- | --- |
| `target_update_interval` 250 / **1.000** / 4.000 | 661,7 / **186,2** / 74,8 | 206,1 / **30,4** / 49,7 |
| `hidden` 64×64 / **256×256** / 512×512 | 58,2 / **186,2** / 558,1 | 0,0 / **30,4** / 279,3 |

Beide Reihen sind **monoton über alle drei Werte** — das ist das eigentliche
Argument, nicht der p-Wert. Bei `hidden_512x512` liegt zudem jeder der fünf
Seeds über jedem Basis-Seed (279 gegen höchstens 266).

Die Deutung ist bei beiden dieselbe und passt zur bekannten Oszillation aus 8.2:
Die Parameter heben nicht die Decke, sondern den Boden. Häufigere
Target-Kopien und ein größeres Netz verhindern, dass die Policy nach einer guten
Phase wieder einbricht. Wer bei 1M Schritten einfach stoppt — und das tut man im
Normalfall —, bekommt dadurch das Drei- bis Vierfache.

**Der Gegenpol bestätigt es:** `target_update_interval` 4.000 ist beim letzten
Checkpoint die **schlechteste** Variante der ganzen Studie (Median 74,8), obwohl
ihr bester Checkpoint mit 464 über der Basis liegt. Ein veraltetes Target macht
gute Phasen nicht unmöglich, nur flüchtig.

**3. Die Explorationsdauer muss zum Reward passen — auf der Zeitachse.** Das war
die offene Frage aus 11.4, und sie hat eine klare Antwort:

| `epsilon_decay_steps` | bis Score 5 | bis Score 10 | Endscore |
| --- | --- | --- | --- |
| 100.000 | **254k** (p = 0,032) | **313k** (p = 0,008) | kein Unterschied |
| **200.000** | **320k** | **395k** | — |
| 400.000 | 467k (p = 0,008) | 518k (p = 0,008) | kein Unterschied |

Alle Schwellenwerte sind Mediane über die fünf Seeds, wie in 8.6. (`significance.py`
rechnet intern mit Mittelwerten und zeigt deshalb leicht andere Zahlen; die
p-Werte stammen ohnehin aus den Einzelwerten je Seed, nicht aus diesen Lagemaßen.)

Streng monoton, an beiden Enden signifikant, und bei „bis 10" überlappen die
Seed-Spannen der drei Werte **überhaupt nicht** (268–321 / 352–421 / 493–550).
Unter `shaped` lässt sich die Explorationsphase also **halbieren, ohne etwas zu
verlieren** — das Lernen wird rund 20 % schneller, die Endqualität bleibt
gleich. Die 200.000 Schritte waren für `legacy` eingestellt (Abschnitt 7), wo
der Agent die erste Röhre erst spät zuverlässig nahm. `shaped` braucht diese
Zeit nicht mehr.

Das ist der vorhergesagte Befund aus 11.4, aber auf der Geschwindigkeits-, nicht
auf der Qualitätsachse. Der hohe Mittelwert von 400k (1.111) ist kein
Gegenargument: Er stammt aus zwei Seeds (2.091 und 2.605), ist zensiert, und
derselbe Arm hat beim letzten Checkpoint den zweitschlechtesten Median der Studie.

**4. Die Lernrate wirkt ausschließlich auf die Geschwindigkeit.** 3e-4 → 304k,
1e-4 → 395k, 5e-5 → 508k bis Score 10; beide Enden signifikant, monoton, und am
Endscore ändert sich nichts (p = 0,69 und 0,31). Die Erwartung aus 11.4, 3e-4
werde instabil, ist **widerlegt** — die höhere Rate lernt schneller und endet
gleich gut. Ein Hinweis, dass in dieser Umgebung noch Luft nach oben ist; die
Studie hat 1e-3 nicht getestet.

**5. Der Wert von n-step hängt am Reward-Schema.** Unter `legacy` war n-step in
der Ablation der **größte Einzelbeitrag** (8.5: Verlauf 38 gegen 15 ohne).
Unter `shaped` ist n = 1 nur noch geringfügig langsamer (425k gegen 395k,
p = 0,032) und sonst nicht zu unterscheiden.

Das ist der schönste Querbezug zwischen den beiden Studien: n-step transportiert
Belohnung über die ~50 Frames Vorlauf bis zur ersten Röhre (8.1) — aber genau
diese Lücke füllt das Shaping ohnehin, indem es jeden Schritt in Richtung
Lückenmitte sofort bewertet. **Zwei Mechanismen für dasselbe Problem; wer einen
hat, braucht den anderen kaum noch.** Die Hypothese aus 11.4 hat das
vorhergesagt.

n = 3 bleibt trotzdem die Basis: Es ist in keiner Messung schlechter, und es
hält die Konfiguration über alle Studien hinweg stabil.

**6. `gamma` 0,99 ist bestätigt — durch Abwesenheit.** Weder 0,95 noch 0,999
schlägt die Basis in irgendeiner Messung. 0,999 hat den niedrigsten Verlauf der
ganzen Studie (48,1 gegen 57,3), 0,95 ist langsamer (432k gegen 395k, nicht
signifikant). Kein Ergebnis ist hier ein Ergebnis: Der Standardwert ist gut
gewählt und der Shaping-Diskont folgt ihm korrekt (6.6).

**7. `hidden` 64×64 ist der einzige klare Verlierer.** Median 58 statt 192,
ein Seed lernt in 1M Schritten **überhaupt nichts** (Score 0,0, erreicht Score 10
nie), die Flap-Rate liegt mit 0,095 als einzige deutlich über allen anderen
(0,061 – 0,079). Damit ist die Hoffnung aus 11.4 erledigt, mit einem kleineren
Netz alle künftigen Experimente zu beschleunigen. 12 Eingangsdimensionen heißt
eben nicht, dass 64 Neuronen reichen — die Q-Funktion, nicht die Eingabe, ist
das Komplizierte.

**8. Das große Netz kostet das Doppelte.** `hidden_512x512` brauchte 137,8 min
je Lauf, die 256×256-Läufe derselben Belegung 68,4 min. Das ist der Preis für
den Stabilitätsgewinn aus Befund 2 und die einzige Zahl der Studie, bei der
Aufwand gegen Nutzen abzuwägen ist.

#### Drei Grenzen der Messung

**Die Zensur ist zurück.** Bei 200.000 Frames liefen acht Lauf-Messungen ins
Limit, die schlimmste (`epsilon_decay_steps_400000_seed1`) in 16,7 % der
Episoden. Deren Scores sind Untergrenzen, in der Tabelle mit \* markiert. Die
Vorhersage in 12.7 („Parameter-Studie, erwartet: keine Zensur") war **falsch** —
die getunten Agenten sind deutlich besser geworden als die der Reward-Studie.
Für den finalen Lauf gilt deshalb 500.000 (12.4), und die Prüfung bleibt:
`truncation_rate` muss 0 sein.

**Der Greedy-Verlauf trennt nur noch nach unten.** Er erkennt die Verlierer
weiterhin zuverlässig — `hidden_64x64` 28,8, `learning_rate_5em05` 44,1,
`gamma_0p999` 48,1. Aber die oberen neun Varianten liegen alle zwischen 53,1 und
62,5, dicht unter der Decke des Trainingslimits von ~79 Röhren (6.4), und dort
trennt er nichts mehr: Die beiden Gewinner aus Befund 2 liegen mit 57,7 und 60,9
neben der Basis mit 57,3. In der Ablation war diese Messung das stärkste
Werkzeug (9,7 bis 38,5 bei `legacy`, weit unter der Decke). Unter `shaped` ist
sie nur noch ein Verlierer-Detektor — eine Eigenschaft gedeckelter Metriken, kein
Fehler: Wo alle Varianten gut sind, misst eine gedeckelte Skala nichts mehr.

**Die Lerngeschwindigkeit war die einzig verlässliche Messung.** Ihre
Seed-Spannen sind eng (jede Variante innerhalb von ~10 %), während die
Endscore-Spannen um den Faktor 6 bis 20 streuen. Jeder belastbare Befund dieser
Studie außer Nr. 2 kommt aus `steps_to_<n>`. Das bestätigt 10.1 nachdrücklich:
`steps_to_<n>` ist nicht nur die gemeinsame Währung über die vier Verfahren,
sondern in dieser Umgebung schlicht die genauere Messung.

#### Reproduzierbarkeit — zum zweiten Mal kostenlos bestätigt

Die `baseline`-Läufe dieser Studie haben dieselbe Trainingskonfiguration wie die
`shaped`-Läufe der Reward-Studie (8.6). Ihre `train.csv` sind **Byte für Byte
identisch**, für alle fünf Seeds — geprüft mit `cmp`.

Das wiegt mehr als die erste Bestätigung in 8.6, denn zwischen den beiden Läufen
liegen ein **Refactoring** des Pakets (Commit `3d7ae0f`, gemeinsame
`rollout`-Funktion und CLI-Definitionen) und ein **Stromausfall** mit
anschließendem Neustart. Beides hat an keinem einzigen Trainingsschritt etwas
geändert. Die einzigen Unterschiede in der `config.json` sind
`eval_max_episode_steps` (50.000 gegen 200.000, betrifft nur das Messen) und
`reward_overrides` (`None` gegen `{}`, eine Schema-Änderung aus dem Refactoring).

#### Zwei der fünf Schwellen sind tot

`steps_to_25` und `steps_to_50` haben in **keinem einzigen der 140 Läufe** aller
Studien je einen Wert geliefert — auch nicht im finalen Lauf (8.9). Der Grund ist die Restexploration: Bei
ε = 0,01 und rund 37,7 Frames je Röhre braucht Score 25 etwa 940 Frames, in denen
im Mittel neun Zufallsaktionen fallen — eine genügt zum Tod (8.2). Der höchste
gleitende Trainings-Score, den überhaupt ein Lauf erreicht hat, liegt bei 17,8;
der Median über die 65 Läufe der Parameter-Studie bei 13,2.

Nutzbar sind damit nur `steps_to_1`, `steps_to_5` und `steps_to_10`. Das ist
kein Schaden — die drei genügen und haben in dieser Studie die Arbeit gemacht —
aber es hat eine Folge für den Verfahrensvergleich, siehe 10.1.

#### Abgleich mit den Hypothesen aus 11.4

| Hypothese vor den Läufen | Ergebnis |
| --- | --- |
| `gamma` 0,99 am besten | **bestätigt** — kein Wert schlägt sie |
| `learning_rate` 1e-4 am besten, 3e-4 instabil | **widerlegt** — 3e-4 lernt schneller und endet gleich gut |
| `n_step` 3 oder 5 am besten | **bestätigt** für 3; 5 ist tendenziell langsamer |
| unter `shaped` ist n-step weniger wichtig als unter `legacy` | **bestätigt** — größter Einzelbeitrag dort, marginal hier |
| `target_update_interval`: mittlerer Wert am besten | **widerlegt** — 250 ist besser, und zwar monoton |
| `epsilon_decay_steps`: 100k als Beleg, dass Exploration zum Reward passen muss | **bestätigt**, auf der Geschwindigkeitsachse (20 % schneller, gleiche Qualität) |
| `hidden`: wenig Unterschied, 64×64 hält mit | **widerlegt in beide Richtungen** — 64×64 fällt stark ab, 512×512 gewinnt deutlich |
| Alle Varianten gleichauf ⇒ Standardwerte sind gut genug | **teilweise** — am Endscore ja, an Stabilität und Tempo nein |

Bemerkenswert: Von acht Vorhersagen sind vier widerlegt. Die beiden Gewinner
(`target_update_interval` 250, `hidden` 512×512) waren beide **nicht**
vorhergesagt, und beide wurden ursprünglich als die zwei *unwichtigsten*
Parameter aus dem ersten 45-Lauf-Durchgang ausgeschlossen (12.3). Sie wurden nur
deshalb gefunden, weil das Raster später vervollständigt wurde.

#### Konsequenz für den finalen Lauf

> ⚠️ **Nachtrag nach 8.9:** Diese Entscheidung wurde getroffen, der finale Lauf
> ist gelaufen — und er hat sie **verworfen**. Die Konfiguration des Projekts
> bleibt die aus Abschnitt 7 (256×256, Target 1.000). Der folgende Absatz
> dokumentiert die Begründung, wie sie **vor** dem Lauf lautete.

Übernommen werden `hidden` 512×512 und `target_update_interval` 250 — die
beiden Änderungen, die in mehreren Messungen **und** monoton über drei Werte
bestehen. Nicht übernommen werden die reinen Geschwindigkeitsgewinne
(`epsilon_decay_steps` 100.000, `learning_rate` 3e-4): Bei einem festen Budget
von 1.000.000 Schritten ist schneller Lernen kein Qualitätsgewinn, und jede
weitere Änderung vergrößert das Risiko einer unerkannten Wechselwirkung.

Denn das ist die Grenze des Aufbaus „ein Parameter je Variante": Dass die beiden
Gewinner **zusammen** wirken, ist nicht gemessen. Beide verbessern dieselbe
Größe — die Stabilität — und könnten sich überschneiden. Der finale Lauf prüft
das deshalb mit drei Armen (12.4).

#### Eine Altlast, die diese Studie sichtbar macht

Befund 5 sagt: Der Wert von n-step hängt am Reward-Schema. Damit steht eine
Frage im Raum, die bisher niemand gestellt hat — denn **die gesamte
Algorithmus-Konfiguration wurde unter `legacy` ausgewählt** (Ablation, 8.5) und
läuft seit der Reward-Studie unter `shaped`.

Das ist nicht bloß formal. Die Begründungen aus Abschnitt 6 und 11.1 stützen sich
ausdrücklich auf den **dichten Überlebensbonus**:

| Baustein | Begründung | Gilt sie unter `shaped`? |
| --- | --- | --- |
| Double DQN | „Der dichte Alive-Reward macht die Q-Werte groß, die Überschätzung durch den Max-Operator ist deshalb relevant" (11.1) | **Nein** — `shaped` hat `alive = 0,0` |
| Huber-Loss (6.1) | „0,1 ÷ (1 − 0,99) = 10, MSE plus große TD-Fehler zerlegt das Netz" | **Nein** — dieselbe Rechnung ergibt unter `shaped` einen viel kleineren Wert |
| Dueling | brachte schon unter `legacy` nichts Messbares (8.5), wurde nur behalten, weil es nicht schadet | unverändert fraglich |
| n-step 3 | größter Einzelbeitrag unter `legacy` | **widerlegt für `shaped`** (Befund 5) |

Bei n-step ist das messbar eingetreten. Bei Double DQN und Huber-Loss ist es
bisher nur ein Argument: Beide *schaden* nach allem, was vorliegt, nicht, und
Huber-Loss ist ohnehin die konservative Wahl. Aber die Sätze in 6.1 und 11.1
begründen sie mit einer Größe, die im aktuell genutzten Schema **null** ist.

**Was daraus folgt — und was nicht.** Der finale Lauf wird deswegen nicht
verschoben: Es gibt keinen Hinweis, dass einer der Bausteine unter `shaped`
schadet, und eine Ablation unter `shaped` (3 Varianten × 5 Seeds ≈ 2,5 Stunden)
würde die Konfiguration bestenfalls vereinfachen, nicht verbessern. Für die
Präsentation gehört der Punkt aber genannt: *„Unsere Bausteine haben wir unter
dem alten Belohnungsschema ausgewählt; für n-step konnten wir zeigen, dass diese
Wahl schemaabhängig ist — für die übrigen haben wir es nicht nachgeprüft."*
Das ist ehrlicher und interessanter als die Behauptung, die Konfiguration sei
durchgängig validiert.

Der allgemeine Satz dahinter ist das eigentliche Ergebnis der drei Studien
zusammen: **Algorithmus-Bausteine und Hyperparameter sind nicht unabhängig vom
Belohnungsschema zu wählen.** Wer das Schema wechselt, muss beides neu prüfen.
Genau deshalb darf die hier gefundene Konfiguration auch nicht ungeprüft in einen
`legacy`-Vergleich übernommen werden (12.8).

### 8.8 Durchsatz

| Aufbau | pro Lauf | gesamt |
| --- | --- | --- |
| ein Lauf allein | ~480 Schritte/s | ~480 Schritte/s |
| 9 Läufe parallel | ~192 Schritte/s | ~1.730 – 2.100 Schritte/s |

Gemessen auf 14 physischen Kernen. Parallelisierung bringt also gut den
vierfachen Gesamtdurchsatz — für dieselbe Zeit, die der einzelne 2M-Lauf
brauchte, bekommt man neun Läufe über je 300.000 Schritte.

---

### 8.9 Finaler Lauf auf neuen Seeds — das Tuning hält nicht

3 Arme × Seeds 100 – 104 = **15 Läufe**, je 1.000.000 Schritte, Reward `shaped`,
Messlimit 500.000 Frames. Aufbau und Begründung in 12.4. Endmessung mit
`summarize.py`: bester Checkpoint über 50 gepaarte Episoden, letzter über 30
(Evaluations-Seed 90.000). Grafik: `runs/final_dqn/final.png`.

Die Frage, für die dieser Lauf gebaut wurde: **Hält der Tuning-Gewinn aus 8.7 auf
Seeds, die bei der Auswahl keine Rolle gespielt haben?**

Die Antwort ist nein.

#### Ergebnis

| Arm | `hidden` | Target | bester CP Ø | Median | letzter CP Ø | **Median** | schlecht. Seed | bis 10 | min/Lauf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`baseline`** | 256×256 | 1.000 | 440,3 ± 351,8 | 359,6 | **488,7 ± 408,7** | **504,5** | **48,3** | 380k | **42,4** |
| `hidden_256x256` | 256×256 | 250 | 380,0 ± 151,5 | 384,0 | 248,8 ± 93,0 | 228,3 | 142,0 | 372k | 75,8 |
| `hidden_512x512` | 512×512 | 250 | 930,8 ± 1.415,5\* | 289,2 | 259,3 ± 188,5 | 289,2 | 13,6 | **312k** | 149,5 |
| Zufall | — | — | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | nie | — |

\* `hidden_512x512_seed102` lief beim besten Checkpoint in 3,3 % der Episoden ins
500.000-Frame-Limit; sein Wert (3.447,7) ist eine Untergrenze und trägt den
Mittelwert des Arms fast allein. **Beim letzten Checkpoint ist die
`truncation_rate` in allen drei Armen 0,000** — die entscheidende Messung ist
zensurfrei.

Die ungetunte Basis gewinnt beim letzten Checkpoint, auf dem Median um Faktor
1,8. Die Einzelwerte:

| Arm | Seeds (letzter Checkpoint) |
| --- | --- |
| `baseline` | 48 · 223 · **504** · 547 · 1.121 |
| `hidden_256x256` | 142 · 198 · **228** · 292 · 384 |
| `hidden_512x512` | 14 · 163 · **289** · 309 · 522 |

#### Signifikanz gegen `baseline`

```powershell
python -m flappy_bird_gymnasium.dqn.significance runs/final_dqn --baseline baseline
python -m flappy_bird_gymnasium.dqn.significance runs/final_dqn --baseline baseline --file evaluations_latest.csv
```

| Variante | bester CP | letzter CP | bis 5 | bis 10 |
| --- | --- | --- | --- | --- |
| `hidden_512x512` | 1,0 | 0,42 | 0,42 | **0,032** (schneller) |
| `hidden_256x256` | 1,0 | 0,42 | 1,0 | 0,84 |

p = 0,42 heißt: Das Tuning ist auch nicht nachweisbar *schlechter*. Nachweisbar
**besser** ist es aber in keiner einzigen Score-Messung — und genau das war die
Behauptung.

#### Der Befund aus 8.7 kehrt sich um

Verglichen wird dieselbe Messung — letzter Checkpoint, `summarize.py`, 30
gepaarte Episoden — einmal auf den Auswahl-Seeds, einmal auf den neuen.

| Form des Befunds | Seeds 0 – 4 (8.7) | Seeds 100 – 104 | |
| --- | --- | --- | --- |
| Median getunt / Basis | 558,1 / 186,2 = **3,00** | 289,2 / 504,5 = **0,57** | umgekehrt |
| Ø getunt / Basis | 783,2 / 173,2 = **4,52** | 259,3 / 488,7 = **0,53** | umgekehrt |
| schlechtester Seed | 279,3 gegen 30,4 | **13,6 gegen 48,3** | umgekehrt |
| p-Wert | **0,008** | 0,42 | weg |

Auch das Nebenargument fällt: Die Flap-Rate der getunten Policy liegt beim
letzten Checkpoint bei 0,092 gegen 0,068 bei der Basis. Nach 8.5, Befund 7
flattern *bessere* Varianten sparsamer.

#### Warum: es gab nichts zu reparieren

Das Tuning sollte den Einbruch zwischen bestem und letztem Checkpoint verhindern
— den „Boden anheben" (8.7, Befund 2). Auf den neuen Seeds tritt dieser Einbruch
bei der Basis gar nicht auf:

| | bester CP Ø | letzter CP Ø | |
| --- | --- | --- | --- |
| `baseline`, Seeds 0 – 4 (8.7) | 442,2 | 173,2 | **−61 %** |
| `baseline`, Seeds 100 – 104 | 440,3 | 488,7 | **+11 %** |

Die Basis-Konfiguration ist auf beiden Seed-Sätzen beim besten Checkpoint
praktisch identisch (442,2 gegen 440,3 — eine bemerkenswert genaue Replikation).
Was sich unterscheidet, ist allein, ob sie bei 1M Schritten gerade in einer guten
oder einer schlechten Phase steht. **Der Einbruch war eine Eigenschaft der Seeds
0 – 4, nicht der Konfiguration.** Das Tuning hat ein Problem behoben, das nur in
der Stichprobe existierte, auf der es ausgewählt wurde.

Das ist Überanpassung an den Auswahldatensatz, im Lehrbuchfall. Bei zwölf
Varianten gegen dieselbe Basis und 0,008 als kleinstmöglichem p-Wert ist ein
Treffer auf diesem Niveau genau das, was der Zufall liefert — 12.4 hat den Fall
vor dem Lauf beschrieben, und er ist eingetreten.

#### Was überlebt

**Die Lerngeschwindigkeit des großen Netzes — als Einziges.** `steps_to_10`
312k gegen 380k (Mediane), p = 0,032. In 8.7 waren es 331k gegen 395k bei
p = 0,008: gleiche Richtung, gleiche Größenordnung, zweimal unabhängig gemessen
auf getrennten Seed-Sätzen. Das ist der einzige replizierte Tuning-Effekt des
Projekts.

Er rechtfertigt das Netz trotzdem nicht: **149,5 gegen 75,8 Minuten je Lauf** —
beide Arme aus demselben Aufruf, also nach 9.8 vergleichbar. 18 % weniger
Schritte für 100 % mehr Rechenzeit, bei gleichem Endergebnis. Auf einem
Wandzeit-Budget (10.4) verliert 512×512.

`target_update_interval` 250 zeigt in **keiner** Messung mehr etwas: Score
p = 0,42, `steps_to_10` p = 0,84, Median unter der Basis.

#### Konsequenz

**Die Konfiguration aus Abschnitt 7 bleibt unverändert die des Projekts.**
`hidden` 256×256 und `target_update_interval` 1.000 werden nicht ersetzt. Der
finale Agent ist die `baseline`-Konfiguration unter Reward `shaped` — sie ist
zugleich die schnellste (42,4 min gegen 149,5) und die einfachste.

Die Endzahl der Präsentation, gemessen auf Seeds, die an keiner Auswahl beteiligt
waren, zensurfrei:

> **Ø 488,7 ± 408,7 Röhren**, Median 504,5, Spanne 48,3 – 1.121,1, über 5 Seeds
> beim letzten Checkpoint. Bester Checkpoint: Ø 440,3 ± 351,8, Median 359,6.
> Zufallsreferenz: 0,00.

Der Mittelwert ist hier die ehrlichere Zahl als in den Vorstudien, weil die
Verteilung weniger schief ist — trotzdem gehört die Streuung immer dazu.

#### Was daraus methodisch folgt

**Monotonie über drei Werte hat eine Überanpassung nicht verhindert.** Das war
in 8.7 das tragende Argument („nicht der p-Wert, sondern die Monotonie"), und es
war nicht falsch — aber es war nicht ausreichend. Beide Parameter waren monoton,
in mehreren Messungen konsistent, und beide halten auf neuen Seeds nicht.

Der einzige Schutz, der funktioniert hat, ist der gehaltene Seed-Satz selbst
(Regel 6). Er hat aus einer Behauptung, die in die Präsentation gegangen wäre,
ein geprüftes — und verworfenes — Ergebnis gemacht. Das ist der stärkste
methodische Befund des Projekts und gehört in die Präsentation.

#### Abgleich mit den Hypothesen aus 12.4

| Erwartung vor dem Lauf | Ergebnis |
| --- | --- |
| Der Tuning-Gewinn hält auf neuen Seeds | **widerlegt** — er kehrt sich um |
| Fällt er negativ aus, war 8.7 ein Zufallstreffer aus zwölf Vergleichen | **eingetreten**, wie in 12.4 beschrieben |
| Liegen `hidden_256x256` und `hidden_512x512` gleichauf, kommt alles von Target 250 | **gegenstandslos** — beide liegen unter der Basis (p = 0,91 untereinander) |
| Das doppelt so teure Netz lohnt sich nicht | **bestätigt**, aus anderem Grund als erwartet |

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

### 9.8 Die Wandzeit einer Variante war kein Messwert

In der Parameter-Studie steht `epsilon_decay_steps_400000` mit 50,3 Minuten je
Lauf da, die Basis mit 101,8 — scheinbar doppelt so schnell, bei identischer
Netzgröße und identischer Schrittzahl. Das ist reine Belegung: 45 Läufe auf 13
Prozessen sind drei volle Durchgänge plus einen mit nur sechs Läufen, und diese
sechs hatten die Maschine fast für sich. Alle fünf 400k-Seeds landeten dort
(49,3 · 49,4 · 50,3 · 50,8 · 51,7 min), bei `epsilon_decay_steps_100000` erwischte
es genau einen Seed (52,6 gegen 97,1 · 97,5 · 97,5 · 98,8).

Die Spalte `minutes` misst also, **wann** ein Lauf an der Reihe war, nicht was er
kostet. Vergleichbar ist sie nur zwischen Läufen aus demselben Durchgang — die
einzige belastbare Aussage der Studie ist deshalb 137,8 gegen 68,4 Minuten für
512×512 gegen 256×256, weil beide Arme in ihrem eigenen 10-Lauf-Aufruf liefen.

*Folge:* Wandzeit nie aus einer parallel gefahrenen Studie als Eigenschaft einer
Variante berichten. Wer Rechenkosten vergleichen will, misst sie in einem
eigenen Lauf bei fester Belegung — oder vergleicht nur innerhalb eines
Durchgangs. Für die Budget-Frage aus 10.4 ist das unmittelbar relevant: Eine
Wandzeit-Definition über die vier Verfahren ist nur dann fair, wenn alle vier
unter derselben Belegung gemessen werden.

### 9.9 Die Zwischenmetrik wuchs nicht mit den Agenten

`quick_eval` misst bewusst am **Trainings**limit von 3.000 Frames (≈ 79 Röhren),
damit die Zwischenevaluation nicht teurer wird als das Training selbst. Das war
in der Ablation richtig: Dort lag der Greedy-Verlauf zwischen 9,7 und 38,5, weit
unter der Decke, und war das **stärkste** Werkzeug der Studie.

Mit besseren Agenten kippte das. Über alle Läufe von Reward-, Parameter- und
finaler Studie liegt die `truncation_rate` bei der Checkpoint-Auswahl zwischen
0,26 und 0,90 — die Hälfte bis neun Zehntel aller Evaluationsepisoden werden
abgeschnitten. Zwei Folgen:

1. **Der Greedy-Verlauf wurde vom Trennwerkzeug zum Verlierer-Detektor.** In 8.7
   liegen die oberen neun Varianten zwischen 53,1 und 62,5 — er erkennt
   `hidden_64x64` (28,8) zuverlässig, unterscheidet oben aber nichts mehr. Das
   steht in 8.7 bereits als „Grenze der Messung".
2. **Die Auswahl von `best.pt` verlor ihre Auflösung.** Der Checkpoint wird über
   den Mittelwert der letzten drei Evaluationen gewählt — eine bewusste und
   richtige Entscheidung gegen Rausch-Auswahl, im Code begründet. Aber Glättung
   hilft gegen Rauschen, nicht gegen eine Decke: Wenn alle späten Checkpoints
   79 melden, ist keiner mehr als der beste erkennbar. Eine Policy, die 300
   Röhren wert ist, und eine, die 3.000 wert ist, sehen identisch aus.

Das erklärt nachträglich, warum die Spalte „bester Checkpoint" in der
Parameter-Studie **keine einzige** Variante trennen konnte (8.7, Befund 1) und
warum ihre Seed-Spannen um Faktor 6 bis 20 streuen: Was dort verglichen wird,
ist nicht der beste Checkpoint einer Variante, sondern ein beliebiger später.

**Wichtig für die Einordnung:** Das erzeugt **Varianz, keine Verzerrung**. Die
Auswahl trifft jede Variante gleich, und die Messung selbst (`summarize.py`)
läuft am hohen Limit und ist korrekt. Die Vergleiche bleiben fair — nur sind die
Fehlerbalken größer, als die Spaltenüberschrift vermuten lässt. Betroffen ist
allein die Best-Checkpoint-Spalte; `steps_to_<n>` wird auf Trainingsepisoden
gemessen, wo der Deckel bei ε = 0,01 nie greift (höchster je erreichter
gleitender Trainings-Score: 17,8), und ist vollständig unberührt.

*Folge:* `quick_eval` braucht ein eigenes, höheres Limit — 15.000 bis 20.000
Frames hätten Kopffreiheit bis 400 bzw. 530 Röhren gegeben und kosten nur bei
den wenigen starken Läufen überhaupt etwas. Und allgemeiner: **Ein Messlimit ist
eine Annahme über das Können des Agenten. Wird der Agent besser, muss das Limit
mitwachsen — sonst misst man irgendwann nur noch das Limit.** Dieselbe Lehre
steht in 9.3 und 12.7 für das *Mess*limit; sie gilt für die Zwischenevaluation
genauso, und dort ist sie zweimal übersehen worden.

---

## 10. Methodik für die Endauswertung

Vier Lernverfahren sollen verglichen werden: **PPO, DQN, Q-Learning und CNN.**

### 10.1 Welche Zahl wofür

| Metrik | Aussage | Grenze |
| --- | --- | --- |
| **Score** | was die fertige Policy kann | wird vom Frame-Limit zensiert |
| **`steps_to_<n>`** | wie schnell gelernt wurde | nicht zensierbar, aber an die Explorationsrate gekoppelt — siehe unten |
| **Return** | das Optimierungsziel des Agenten | **nur innerhalb desselben Reward-Schemas vergleichbar** |

`steps_to_<n>` ist die Zahl der Umgebungsschritte, bis der gleitende Score über
20 Episoden ein Niveau erstmals hielt. Der gleitende Mittelwert verhindert, dass
eine einzelne Glücksepisode als „erreicht" zählt.

**Einschränkung, die vor dem Verfahrensvergleich geklärt werden muss.** Diese
Metrik wird auf den **Trainings**-Episoden gemessen, also mit eingeschalteter
Exploration. Für DQN heißt das ε = 0,01, und das deckelt den Trainings-Score
strukturell: In 140 Läufen hat **kein einziger** je `steps_to_25` erreicht
(8.7, 8.9). Zwei Konsequenzen für den Vergleich mit PPO, Q-Learning und CNN:

1. **Nur Schwellen verwenden, die alle vier Verfahren erreichen können.** Für
   DQN sind das 1, 5 und 10. Höhere Schwellen sind keine „schwierigeren
   Messpunkte", sondern für DQN schlicht undefiniert.
2. **Die Deckelung ist je Verfahren verschieden.** PPO sampelt aus einer
   stochastischen Policy, Q-Learning hat seinen eigenen ε-Zeitplan. Wie stark
   der Trainings-Score gedeckelt ist, hängt also vom Verfahren ab — ein
   Unterschied in `steps_to_10` kann daher auch ein Unterschied in der
   Explorationsrate sein statt in der Lerngeschwindigkeit.

Sauber wird die Metrik erst, wenn sie aus den **Greedy**-Zwischenmessungen käme
statt aus den Trainingsepisoden. Das ist im DQN-Paket nicht umgesetzt (`eval.csv`
liegt nur alle 50.000 Schritte vor, das ist für eine Schwelle zu grob). Solange
das so bleibt, gehört zu jeder `steps_to`-Zahl im Vergleich der Satz, mit welcher
Explorationsrate sie gemessen wurde.

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

### 11.4 Parameter-Studie (Aufbau in 12.3, Ergebnisse in 8.7)

Basis ist `shaped` mit der Konfiguration aus Abschnitt 7. Der Abgleich steht am
Ende von 8.7: **vier der acht Vorhersagen sind widerlegt**, und beide Gewinner
wurden nicht vorhergesagt.

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

### 12.3 Parameter-Studie — gelaufen, Ergebnisse in 8.7

Sucht die beste Einstellung für den finalen Lauf. Hyperparameter, jeder für sich
um eine gemeinsame Basis herum variiert.

**So gelaufen: alle sechs Parameter, 65 Läufe, in drei Aufrufen.** Zuerst die
vier unten als „ja" markierten (45 Läufe), dann `--params hidden` (10), dann der
Rest des Rasters (10). Der letzte Aufruf war der Befehl ohne `--params`: Er
überspringt alles Fertige, rechnet nur `target_update_interval` und schreibt
`study.json` mit allen 65 Einträgen neu.

```powershell
# so ist es gelaufen (drei Aufrufe, der letzte holt den Rest)
python -m flappy_bird_gymnasium.dqn.experiments params `
    --seeds 5 --total-steps 1000000 --reward shaped `
    --params learning_rate gamma n_step epsilon_decay_steps `
    --eval-interval 50000 --eval-episodes 10 --eval-max-episode-steps 200000 --workers 13
python -m flappy_bird_gymnasium.dqn.experiments params ... --params hidden
python -m flappy_bird_gymnasium.dqn.experiments params ...        # ohne --params
```

**Dass die Aufteilung folgenlos ist, ist kein Zufall, sondern Aufbau:** Jeder
Lauf trägt seine eigene `config.json`, die Basis ist über alle Aufrufe dieselbe,
und `experiments.py` überspringt einen Lauf nur bei identischer Konfiguration
(Abschnitt 14). Die 65 Läufe sind deshalb auswertbar, als wären sie in einem
Durchgang entstanden. Was die Aufteilung **doch** verfälscht hat, ist die
Wandzeit — siehe 9.8.

**65 Läufe** (13 Varianten × 5 Seeds), rund **8,5 Stunden**, etwa 150 MB.

**Im Nachhinein die wichtigste Lehre dieser Studie:** Die vier „wichtigsten"
Parameter wurden vorab ausgewählt, um Zeit zu sparen — und **beide Gewinner
standen in der gestrichenen Hälfte** (8.7, Hypothesen-Abgleich). Hätte der Lauf
bei 45 aufgehört, wäre das Ergebnis „die Standardkonfiguration ist schon gut"
gewesen. Bei einem Raster, dessen Kosten in der Zahl der Parameter nur *linear*
wachsen, lohnt das Streichen selten.

Die Zeit ergibt sich aus der Belegung: 13 Läufe laufen gleichzeitig, jeder
braucht rund 100 Minuten, ein Durchgang dauert also etwa 1,7 Stunden. Die ersten
45 Läufe sind drei volle Durchgänge plus einen mit nur 6 Läufen — der letzte ist
schneller, weil sich weniger Läufe die Kerne teilen. Jeder weitere Parameter
kostet 10 Läufe, also rund 1,5 Stunden; die beiden nachgezogenen zusammen knapp
2,5 Stunden, weil `hidden` 512×512 doppelt so lange rechnet.

Genau diese ungleiche Belegung ist es, die die `minutes`-Spalte unbrauchbar
macht — siehe 9.8.

| Parameter | Basis | getestet | Erwartung | Befund (8.7) |
| --- | --- | --- | --- | --- |
| `learning_rate` | 1e-4 | 5e-5, 3e-4 | 1e-4 am besten; 3e-4 vermutlich instabil | nur Tempo, monoton; 3e-4 nicht instabil |
| `gamma` | 0,99 | 0,95, 0,999 | 0,99 am besten; 0,95 zu kurzsichtig für 50 Frames Vorlauf, 0,999 instabil | 0,99 bestätigt, kein Wert schlägt sie |
| `n_step` | 3 | 1, 5 | 3 oder 5; n = 1 war unter `legacy` klar schlechter (8.5) | 3 bleibt; unter `shaped` kaum noch Wirkung |
| `epsilon_decay_steps` | 200.000 | 100.000, 400.000 | offen — `shaped` lernt schneller, vielleicht genügt weniger Exploration | 100k lernt 20 % schneller, gleiche Qualität |
| `target_update_interval` | 1.000 | 250, 4.000 | mittlerer Wert am besten | **250 gewinnt**, monoton — Stabilität |
| `hidden` | 256×256 | 64×64, 512×512 | wenig Unterschied; wenn 64×64 mithält, ist das ein Argument für kleinere Netze | **512×512 gewinnt**, 64×64 fällt stark ab |

Alle sechs Parameter sind im Raster hinterlegt; ohne `--params` laufen sie alle.
Ursprünglich waren nur die ersten vier geplant — die mit der größten erwarteten
Wirkung: Lernrate und gamma als klassische Stellschrauben, n-step als stärkster
Einzelbaustein der Ablation, die Explorationsdauer als offene Frage zum
schnelleren Reward. **Genau diese Auswahl war der Fehler:** Die beiden Gewinner
standen in der gestrichenen Hälfte.

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

### 12.4 Finaler Lauf — gelaufen, Ergebnisse in 8.9

Die in 8.7 gewählte Konfiguration auf **neuen Seeds**, in drei Armen. Zwei
Befehle in **dasselbe** `--runs-root` — die Armnamen kollidieren nicht, und nur
so liegen alle drei in einem Verzeichnis und sind gepaart vergleichbar.

```powershell
# Arm 2 und 3: getunt, mit und ohne das grosse Netz
python -m flappy_bird_gymnasium.dqn.experiments sweep `
    --param hidden --values 256x256 512x512 `
    --seeds 5 --seed-offset 100 --total-steps 1000000 --reward shaped `
    --target-update-interval 250 `
    --eval-interval 50000 --eval-episodes 10 --eval-max-episode-steps 500000 `
    --runs-root runs/final_dqn --workers 13

# Arm 1: die ungetunte Basis auf denselben neuen Seeds
python -m flappy_bird_gymnasium.dqn.experiments seeds `
    --seeds 5 --seed-offset 100 --total-steps 1000000 --reward shaped `
    --eval-interval 50000 --eval-episodes 10 --eval-max-episode-steps 500000 `
    --runs-root runs/final_dqn --workers 13
```

**15 Läufe** (3 Arme × Seeds 100 – 104), rund **3 bis 3,5 Stunden**, etwa 50 MB.
Bricht ein Lauf ab: denselben Befehl erneut starten, Fertiges wird übersprungen.

| Arm | `hidden` | `target_update_interval` | Rolle | Ausgang (8.9) |
| --- | --- | --- | --- | --- |
| `baseline` | 256×256 | 1.000 | **Kontrolle**: die ungetunte Konfiguration aus Abschnitt 7 | **gewinnt** |
| `hidden_256x256` | 256×256 | 250 | nur die billige der beiden Änderungen | kein Effekt |
| `hidden_512x512` | 512×512 | 250 | beide Gewinner aus 8.7 | widerlegt |

**Warum drei Arme und nicht einer:** Die beiden Gewinner wurden aus zwölf
Vergleichen auf den Seeds 0 – 4 ausgewählt. Wer so auswählt, findet auch ohne
echten Effekt gelegentlich etwas (8.7, Einordnung der p-Werte). Ein finaler Lauf
*nur* mit der getunten Konfiguration könnte deshalb sagen „unser bester Agent
schafft N Röhren", aber **nicht** „das Tuning hat geholfen" — dafür fehlt die
Vergleichszahl auf Seeds, die bei der Auswahl keine Rolle gespielt haben. Der
`baseline`-Arm kostet 5 Läufe und rund eine Stunde und macht aus einer
ausgewählten Behauptung ein geprüftes Ergebnis.

Der dritte Arm zerlegt den Gewinn zusätzlich in seine zwei Ursachen: Liegen
`hidden_256x256` und `hidden_512x512` gleichauf, kommt alles von
`target_update_interval` 250, und das doppelt so teure Netz (138 gegen 68 min)
lohnt sich nicht. Das ist unmittelbar handlungsrelevant für alle weiteren Läufe.

#### Was hier warum eingestellt ist

| Einstellung | Warum |
| --- | --- |
| `hidden` 512×512 | Gewinner in 8.7: letzter Checkpoint p = 0,008 (jeder Seed über jedem Basis-Seed), Lerngeschwindigkeit p = 0,008, monoton über 64/256/512 |
| `target_update_interval` 250 | Gewinner in 8.7: letzter Checkpoint p = 0,032, Median 662 gegen 186, monoton über 250/1.000/4.000 |
| `epsilon_decay_steps` 200.000 (unverändert) | 100.000 lernt 20 % schneller, endet aber **gleich gut** (8.7). Bei festem Budget ist Tempo kein Qualitätsgewinn — und jede zusätzliche Änderung vergrößert das Risiko einer Wechselwirkung |
| `learning_rate` 1e-4 (unverändert) | dieselbe Begründung: 3e-4 ist schneller, nicht besser |
| `n_step` 3, `gamma` 0,99, `dueling` an | in 8.7 von keiner Alternative geschlagen; hält die Konfiguration über alle Studien vergleichbar |
| Reward `shaped` | schnellstes und bestes Schema (8.6). **Nur für „was kann unser bester DQN-Agent?"** — für den Verfahrensvergleich gilt 12.8 |
| 1.000.000 Schritte | eingespieltes Budget aller Studien. Mehr würde die Vergleichbarkeit mit Ablation, Reward- und Parameter-Studie brechen, bevor die Budget-Frage aus 10.4 entschieden ist |
| Seeds 100 – 104 | siehe unten |
| Messlimit 500.000 | siehe 12.7; in 8.7 hat schon das 200.000er-Limit wieder zensiert |
| `--workers 13` | der eingespielte Wert auf 14 Kernen. 15 Läufe sind ein voller Durchgang plus zwei Nachzügler — die `minutes`-Spalte ist dadurch wieder verzerrt (9.8), die Ergebnisse nicht |

**Warum neue Seeds:** Wer auf den Seeds 0 – 4 die Parameter aussucht und dann auf
denselben Seeds das Ergebnis berichtet, berichtet zu optimistisch — die Wahl hat
sich an den Zufall genau dieser Seeds angepasst. Der finale Lauf misst die
gewählte Konfiguration deshalb auf Seeds, die bei der Auswahl keine Rolle
gespielt haben. Das ist dieselbe Trennung wie zwischen Trainings- und Testdaten.

**Was der dritte Arm zusätzlich leistet:** Der Aufbau „ein Parameter je
Variante" (12.3) kann Wechselwirkungen nicht finden. Die beiden Gewinner
verbessern **dieselbe** Größe — die Stabilität — und könnten sich überschneiden.
Mit drei Armen ist der Gewinn zerlegt: `baseline` → `hidden_256x256` misst den
Beitrag von `target_update_interval` 250, `hidden_256x256` → `hidden_512x512`
den des größeren Netzes. Fallen die letzten beiden zusammen, ist das ein echtes
Ergebnis — und ein Argument, alle weiteren Experimente mit dem kleineren Netz zu
fahren.

#### Auswertung danach

```powershell
python -m flappy_bird_gymnasium.dqn.summarize runs/final_dqn --episodes 50
python -m flappy_bird_gymnasium.dqn.summarize runs/final_dqn --episodes 50 --checkpoint latest.pt
python -m flappy_bird_gymnasium.dqn.significance runs/final_dqn --baseline baseline
python -m flappy_bird_gymnasium.dqn.significance runs/final_dqn --baseline baseline --file evaluations_latest.csv
python -m flappy_bird_gymnasium.dqn.plot runs/final_dqn --study --out runs/final_dqn/final.png
```

`--baseline baseline` ist hier der ungetunte Arm — der Test beantwortet damit
genau die Frage, für die er da ist: **Hält der Tuning-Gewinn auf Seeds, die bei
der Auswahl keine Rolle gespielt haben?** Der letzte Checkpoint ist dabei die
entscheidende Messung, weil der Gewinn in 8.7 dort lag und nicht beim besten.

`--max-episode-steps` wird **nicht** übergeben: Ohne das Flag nimmt `summarize.py`
den Wert aus der `config.json`, und der steht durch den Lauf bereits auf 500.000.
Damit heißen die Dateien schlicht `evaluations.csv` und `evaluations_latest.csv`,
und `significance.py` findet sie ohne `--file`.

**Rechnen mit einer knappen Stunde allein für die Messung.** 50 Episoden bei
einem Agenten um 800 Röhren sind rund 1,5 Millionen Frames je Lauf — die
Evaluation wird hier erstmals teurer als ein spürbarer Teil des Trainings.

**Danach als Erstes `truncation_rate` prüfen.** Ist sie > 0, war auch 500.000
zu wenig und die Endzahl ist eine Untergrenze (Regel 3). Dann mit
`--max-episode-steps 2000000` nachmessen; das erzeugt getrennte Dateien und
lässt die bestehenden unangetastet.

#### Für die Aufnahmen

Die Aufnahme darf aus dem **besten Seed** stammen — eine Aufnahme ist eine
Illustration, kein Messwert. Regel 2 („einen Seed wählt man nicht aus") gilt für
*berichtete Zahlen*: Die Zahl in der Präsentation ist der Mittelwert über die
fünf Seeds mit seiner Streuung, nicht der Wert des Laufs, den man gefilmt hat.
Beides nebeneinander zu zeigen ist ehrlich, solange dabeisteht, was was ist.

Praktischer Hinweis: Bei 800 Röhren dauert eine vollständige Episode rund 30.000
Frames, also **gut 8 Minuten Video bei 60 fps**. Für die Präsentation genügt ein
Ausschnitt — oder ein niedrig gesetztes `--max-episode-steps` beim Aufzeichnen.

### 12.4a Abbildungen und Aufnahme

Beide Ausgaben landen in `docs/`, das in `.gitignore` steht — sie sind aus Code
und Lauf-Logs jederzeit neu erzeugbar.

```powershell
python -m flappy_bird_gymnasium.dqn.figures            # 5 PNG, wenige Sekunden

# Der Einstieg der staerksten Episode -- rund 2 Minuten
python -m flappy_bird_gymnasium.dqn.record `
    runs/final_dqn/hidden_512x512_seed102/best.pt --out docs/dqn_agent_start.gif `
    --seed 90010 --seconds 30 --every 2 --scale 0.75

# Ihr Ende bei 13.273 Roehren -- rund 15 Minuten, weil erst 500.000 Frames
# simuliert werden muessen, bevor aufgezeichnet wird
python -m flappy_bird_gymnasium.dqn.record `
    runs/final_dqn/hidden_512x512_seed102/best.pt --out docs/dqn_agent_13273.gif `
    --seed 90010 --skip-to-pipe 13245 --max-steps 500000 `
    --seconds 60 --every 2 --scale 0.75
```

| Datei | Zeigt | Quelle |
| --- | --- | --- |
| `dqn_ergebnis.png` | die Endzahl mit Seed-Streuung gegen die Zufallsreferenz | 8.9 |
| `dqn_stufen.png` | Lehrbuch-DQN → Bausteine → Reward, in Leistung und Tempo | 8.5, 8.6 |
| `dqn_reward.png` | die 7 Schemata, getrennt nach Strafe aufs Flattern | 8.6 |
| `dqn_ablation.png` | Beitrag der einzelnen Algorithmus-Bausteine | 8.5 |
| `dqn_lernkurve.png` | Trainingsverlauf mit Seed-Band — das Format für den Verfahrensvergleich | 8.9, 8.5 |
| `dqn_agent_start.gif` | 30 s Einstieg der stärksten Episode, Zähler bis 23 | — |
| `dqn_agent_13273.gif` | ihre letzten 36 s, Zähler endet bei **13.273** | — |

`figures.py` hält drei Regeln ein, die aus Abschnitt 9 folgen: **ein Punkt je
Seed** statt Balken mit Fehlerbalken (die Verteilungen sind stark schief),
**letzter statt bester Checkpoint** als Kopfzahl (9.9), und **nie der Return**
zwischen Reward-Schemata (10.1). Keine Zahl ist im Code eingetragen; alle werden
aus den `evaluations*.csv` gelesen.

Die Aufnahmen stammen aus `hidden_512x512_seed102` — dem Lauf aus dem
**verworfenen** Arm, der mit Ø 3.447 Röhren der stärkste Einzel-Checkpoint des
Projekts ist. Das ist kein Widerspruch zu 8.9: Eine Aufnahme ist eine
Illustration, kein Messwert (12.4, letzter Absatz). Die begleitenden `.txt`
schreiben das mit auf, damit die Zuordnung nicht verlorengeht.

**Welche Episode die 13.273 war.** Der berichtete `max_score` ist das Maximum
über die 30 Evaluationsepisoden (Seeds 90.000 – 90.029); welche es war, steht in
keiner Ausgabedatei. Alle 30 einzeln nachgespielt ergibt
`runs/final_dqn/_episode_scan.json`: Es ist **Seed 90.010**, und die Episode ist
nicht zu Ende gegangen — sie lief mit 500.000 Frames ins Messlimit, der Agent
lebte noch. Die 13.273 sind also eine Untergrenze. Der Nachlauf hat nebenbei den
Mittelwert von 3.447,7 exakt reproduziert, ein dritter kostenloser
Reproduzierbarkeitsnachweis.

**Der Punktestand wird von `record.py` selbst ins Bild gezeichnet.** Die
Umgebung blendet ihn im `rgb_array`-Modus bewusst aus
(`flappy_bird_env.render`), und ihre Ziffern-Sprites werden ohne
Alpha-Konvertierung als weißer Block geblittet. Ohne den Zähler wäre der
13.273-Clip von jedem anderen nicht zu unterscheiden — er ist der ganze Punkt
der Aufnahme.

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
| 3 | Parameter-Studie | erledigt, 8.7 |
| 4 | Finaler Lauf auf neuen Seeds, unzensiert ausgewertet | **erledigt, 8.9** — Tuning widerlegt |
| 5 | **Aufnahmen des finalen Agenten** | **als Nächstes**; `rl/record.py` aus dem PPO-Branch übernehmen |
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
| Parameter-Studie | 200.000 | **acht Lauf-Messungen zensiert, schlimmste zu 16,7 %** |
| **Finaler Lauf** | **500.000** | Reserve, siehe unten |

**Diese Tabelle enthält eine widerlegte Vorhersage — sie bleibt bewusst stehen.**
Für die Parameter-Studie stand hier „erwartet: keine Zensur", begründet damit,
dass bei 200.000 Frames in der Reward-Studie keine einzige Episode mehr ins Limit
lief. Gleich im nächsten Absatz stand die Einschränkung: Die beste Episode von
`survival_seed3` dauerte rund 171.000 Frames, also **85 % des Limits**, und ein
Agent, der nach der Parametersuche besser wird, überschreitet das. Genau so kam
es (8.7). Die Warnung war richtig, die Vorhersage falsch — wer eine Reserve von
15 % „ausreichend" nennt, hat keine Reserve.

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

  **Die Hyperparameter-Frage hat sich hier erledigt:** Das Tuning aus 8.7 ist
  auf neuen Seeds widerlegt (8.9), es gibt also keine getunte Konfiguration mehr,
  die man versehentlich mitnehmen könnte. Über alle Schemata hinweg gilt die
  Konfiguration aus Abschnitt 7. Der Vorbehalt aus 8.7, Befund 5, bleibt
  trotzdem gültig und wichtig: **Algorithmus-Bausteine sind schemaabhängig** —
  n-step ist unter `legacy` der größte Einzelbeitrag und unter `shaped` fast
  wirkungslos. Für einen `legacy`-Vergleich ist die unter `legacy` validierte
  Bausteinwahl aus 8.5 die richtige.
- **Für „was kann unser bester DQN-Agent?"** ist `shaped` oder `survival` die
  Antwort. Die berichtete Zahl ist der finale Lauf: **Ø 488,7 ± 408,7 Röhren**
  über fünf gehaltene Seeds (8.9). Der stärkste je gemessene Einzel-Checkpoint
  ist `runs/final_dqn/hidden_512x512_seed102/best.pt` mit Ø 3.447,7 Röhren
  (Median 1.823,5, beste Episode 13.273 — zensiert, also eine Untergrenze). Er
  taugt für **Aufnahmen**, nicht als berichtete Zahl (Regel 2, 12.4).

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
| Wechselwirkung der beiden Parameter-Gewinner | **erledigt** — gegenstandslos, beide Gewinner sind auf neuen Seeds widerlegt (8.9) |
| `quick_eval` misst am Trainingslimit (~79 Röhren) | **offen und folgenreich** — deckelt den Greedy-Verlauf und die Auswahl von `best.pt`, sobald Agenten besser als ~79 Röhren werden (9.9). Fix: eigenes, höheres Limit für `quick_eval` |
| Double DQN, Dueling und Huber-Loss unter `shaped` nie geprüft | offen; unter `legacy` ausgewählt (8.5), Begründungen stützen sich auf den Überlebensbonus, den `shaped` nicht hat (8.7, „Eine Altlast") — eine Ablation unter `shaped` kostet ~2,5 h |
| `steps_to_<n>` aus Greedy- statt Trainingsepisoden | offen; die Metrik ist an die Explorationsrate gekoppelt und damit über Verfahren hinweg nur eingeschränkt vergleichbar (10.1) |
| Lernrate über 3e-4 | offen; 3e-4 war schneller und nicht instabil (8.7), 1e-3 wurde nie getestet |
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
