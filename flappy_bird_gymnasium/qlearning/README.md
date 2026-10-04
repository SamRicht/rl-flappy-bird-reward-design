# Tabellarisches Q-Learning

Das vierte Verfahren im Reward-Vergleich, und das einzige ohne Funktions-
approximation. Statt eines Netzes hält der Agent einen Wert pro Zustand-Aktions-
Paar. Damit wird die **Rasterisierung** der kontinuierlichen Beobachtung in einen
endlichen Zustandsraum selbst zu einem Hyperparameter — der interessanteste hier,
weil DQN, PPO und CNN ihn gar nicht haben.

Code und Docstrings sind englisch, diese Datei ist deutsch — wie im DQN-Paket.

## Einrichtung

Reines NumPy, keine zusätzlichen Abhängigkeiten über die des Pakets hinaus:

```bash
source .venv/bin/activate
pip install -e .
```

## Ein Lauf

```bash
python -m flappy_bird_gymnasium.qlearning.train --run-name q1 --reward legacy
```

Dauert mit den Defaults fünf bis sieben Minuten (23 000–31 000 Schritte/s,
10 Mio. Schritte) und erreicht ungezensiert einen Score um 150 im Mittel bei
einem Median um 100. Für einen schnellen Blick `--total-steps 2000000`, aber
dann mit deutlich schlechterem Ergebnis — siehe „Warum das Budget so groß ist".
Ergebnis in `runs/q1/`:

| Datei | Inhalt |
|---|---|
| `config.json` | die vollständige Konfiguration, exakt die Felder von `QLearningConfig` |
| `train.csv` | eine Zeile je Episode: `step, episode, return, score, length, flap_rate, epsilon, alpha, td_error, coverage` |
| `eval.csv` | eine Zeile je Zwischenmessung: `step` plus alles, was `summarize_rollout` liefert |
| `best.npz` / `latest.npz` | Q-Tabelle, Besuchszähler und die Konfiguration |
| `summary.json` | Kennzahlen des Laufs inkl. `steps_to_<n>`, `coverage`, `n_states` |

Die Messung für einen Bericht kommt **nicht** aus `train.csv`, sondern aus:

```bash
python -m flappy_bird_gymnasium.qlearning.evaluate --checkpoint runs/q1/best.npz --episodes 50
python -m flappy_bird_gymnasium.qlearning.evaluate --checkpoint runs/q1/best.npz --episodes 3 --render
```

## Die Rasterisierung

Aus den zwölf Beobachtungswerten werden drei Größen in Rohpixeln gebildet:

| Größe | Bedeutung | Bereich |
|---|---|---|
| `dx` | Abstand von der rechten Vogelkante zur nächsten Röhre | `[-52, 197]` |
| `dy` | Abstand der Vogelmitte zur Mitte der Lücke | theor. `[-300, 242]`, Masse in `[-90, 110]` |
| `vel` | vertikale Geschwindigkeit | ganzzahlig `[-9, 10]` |

Alles andere ist redundant oder fast konstant: `rot` ist wie `vel` eine Funktion
der Frames seit dem letzten Flattern, die absoluten Koordinaten zählen nur über
ihre Differenz, und der dritte Röhren-Slot ist in ~80 % der Frames der
Off-Screen-Platzhalter.

Gebinnt wird über `np.searchsorted`: `n` Kanten ergeben `n+1` Bins und Werte
außerhalb landen automatisch im äußersten Bin, ohne separates Clipping. Der
Zustandsindex ist die Bin-Kombination in gemischter Radix.

### Die drei Schemata

Auswahl über `--discretizer`:

| Schema | Zustände | Median-Score bei 2M | bei 10M | Abdeckung |
|---|---|---|---|---|
| `uniform` (Default) | 15 360 | 29 | **183** | 0.38 |
| `adaptive` | 10 080 | 53 | 61 | 0.60 |
| `lookahead` | 50 400 | 12 | 56 | 0.30 |

*Vier Seeds je Zelle, `legacy`, gepaarte Messung. Berichtet ist der Median der
Lauf-Mediane — der Mittelwert ist bei dieser Verteilung unbrauchbar, siehe
unten.*

**Die Rangfolge hängt vom Budget ab und kippt zwischen den beiden gemessenen
Punkten.** Bei 2 Mio. Schritten gewinnen die handgesetzten Kanten, weil sich
eine kleinere Tabelle schneller füllt. Bei 10 Mio. hat jede Zelle genug
Besuche, und das feinere gleichmäßige Gitter zieht davon — auf *jedem* Seed
(144/164/202/246 gegen 26/56/67/132, exakter Permutationstest p = 0.029, das
Minimum, das vier Seeds erzeugen können).

Das ist die eigentliche Lehre und nicht „uniform ist besser": eine
Rasterisierung ist nur zusammen mit einem Budget bewertbar. Wer die Zahlen für
einen Bericht braucht, muss beide Achsen nennen — und mehr Seeds fahren, denn
bei sechs Vergleichen gegen dieselbe Referenz wäre nach Bonferroni p < 0.0083
nötig, was mit vier Seeds nicht erreichbar ist.

`adaptive` setzt die Kanten von Hand: fein, wo die Entscheidung kippt, grob, wo
sie es nicht tut. `lookahead` ergänzt die übernächste Röhre und kostet dafür
Faktor 5 an Tabellengröße. Beide bleiben registriert — gerade weil der
Budget-Effekt reproduzierbar sein sollte.

Die Bin-Anzahl von `uniform` ist über `--dx-bins/--dy-bins/--vel-bins`
einstellbar und hat ein ausgeprägtes Optimum:

| Zustände | 5 120 | 15 360 | 25 600 | 44 800 | 80 640 |
|---|---|---|---|---|---|
| Median-Score bei 10M | 42 | **183** | 44 | 40 | 33 |
| Abdeckung | 0.55 | 0.38 | 0.35 | 0.31 | 0.29 |

Ein scharfes Optimum: zu grob wirft tödlich verschiedene Situationen in einen
Topf, zu fein verhungert die Tabelle an Besuchen. Genau diese Kurve ist der
tabellenspezifische Beitrag zum Vergleich — DQN, PPO und CNN haben diese
Stellschraube gar nicht.

`vel_y` nimmt exakt 20 ganzzahlige Werte an — mit `--vel-bins 20` ist es ohne
Aliasing aufgelöst, mehr Bins hinterlassen nur Lücken.

## Exploration: `--q-init` trägt sie, nicht `--epsilon-decay-steps`

Die beiden Regler tun hier nicht dasselbe, und das erklärt, warum längere
ε-Abkühlung nichts bringt. `q_init` exploriert **systematisch**: jede noch
nicht probierte Aktion sieht attraktiv aus, bis sie probiert wurde. ε
exploriert **zufällig** — und eine Zufallsaktion zur falschen Zeit ist in
Flappy Bird tödlich. Ein langes ε-Plateau kauft also vor allem Tode, die über
die Policy nichts aussagen.

Messbar, sechs Seeds unter der Default-Rasterisierung, Schritte bis der
gleitende Score 10 hält (in Mio.):

| `epsilon_decay_steps` | Seeds einzeln | Median | p gegen 3M |
|---|---|---|---|
| 150k | 1.24 / 1.25 / 1.29 / 1.47 / 1.60 / 2.21 | 1.38 | 0.0022 |
| **300k** (Default) | 0.98 / 1.12 / 1.23 / 1.44 / 1.72 / 2.41 | 1.33 | 0.0022 |
| 3M | 3.14 / 3.16 / 3.42 / 3.57 / 3.60 / 3.90 | **3.50** | — |

Die Gruppen trennen vollständig — p = 0.0022 ist das Minimum bei sechs gegen
sechs Seeds und hält auch unter Bonferroni-Korrektur für die drei Vergleiche.
150k und 300k sind ununterscheidbar (p = 0.59), deshalb bleibt der längere der
beiden als der sicherere Default stehen. Ein Vorlauf mit vier Seeds unter
`adaptive` kam zum selben Schluss, es ist also kein Artefakt der
Rasterisierung.

Der **Endscore** zeigt bei genau diesen Läufen **gar nichts** (p = 0.48 und
1.00, Mediane 54.8 / 53.3 / 42.5).

Das ist das beste Argument dafür, `steps_to_<n>` als primäre Kennzahl zu führen
und den Score als sekundäre — auch für DQN, PPO und CNN. Derselbe Datensatz
trennt in der einen Metrik perfekt und in der anderen überhaupt nicht. Wer nur
die Score-Tabelle liest, hätte hier „kein Effekt" geschrieben, obwohl der
Effekt Faktor 2.6 beträgt.

## Der wichtigste Regler: `--q-init`

Mit Abstand der einflussreichste Parameter, und zwar aus einem Grund, der nicht
an Flappy Bird hängt:

| `q_init` | 0.0 | 5.0 | 10.0 | 20.0 |
|---|---|---|---|---|
| Score | 0.8 | 10.7 | **12.3** | 3.1 |

Bei `q_init = 0.0` (pessimistisch) rastet die Greedy-Policy auf diejenige Aktion
ein, die als Erste einen positiven Wert bekommen hat, und probiert die andere nie
wieder — der Reward ist überwiegend positiv, jeder gelernte Wert schlägt also die
uninitialisierte Null. Oberhalb des erreichbaren Return sieht dagegen jede noch
nicht probierte Aktion attraktiv aus und wird einmal getestet („optimistische
Initialisierung").

Der passende Wert hängt am Reward-Schema: `q_init` muss in der Nähe von V*
liegen, dem Return einer perfekten Policy. In der Lernmatrix lag das beste
`q_init` bei etwa 0,8–1,0 × V*, deutlich darunter oder darüber fällt der Score.
V* enthält auch den Röhren-Bonus. Alle ~38 Frames kommt eine Röhre, also ist
`Σ γ^(38k) ≈ 3`:

| Schema | V* | bestes `q_init` |
|---|---|---|
| `legacy`, `additive`, `risk_averse`, `energy` | `0.1/(1-γ) = 10` plus ≈ 0,9 × 3 für die Röhren, also ≈ 12–13 | 10 |
| `survival` | genau `0.1/(1-γ) = 10` | 10 |
| `sparse`, `shaped` | nur die Röhren, ≈ 2–3 | 2,5 |

Der Default 10.0 passt also zu den dichten Schemata, ist für `sparse` und
`shaped` aber viel zu hoch. **Eine Reward-Studie muss `q_init` je Schema an V*
anpassen**, sonst vergleicht sie unabsichtlich auch noch unterschiedlich gut
explorierte Läufe.

## Warum das Budget so groß ist

Der Score ist weder durch die Rasterisierung noch durch die Exploration
begrenzt, sondern schlicht durch die Konvergenz. Median über drei Seeds,
40 Episoden gegen das hohe Frame-Limit gemessen:

| Budget | Schrittweite | Median-Score | max |
|---|---|---|---|
| 2 Mio. | konstant 0.3 | 45 | 255 |
| 2 Mio. | 0.3 → 0.02 | 41 | 215 |
| 10 Mio. | konstant 0.3 | 66 | 530 |
| 10 Mio. | 0.3 → 0.02 | **105** | 530 |
| 10 Mio. | 0.3 → 0.005 | 123 | 530 |

Eine konstante Schrittweite lässt die Tabelle nie zur Ruhe kommen: die
Umgebung ist stochastisch, jeder Besuch zieht einen Eintrag zu einer anderen
Stichprobe. Abkühlen hilft aber **erst, wenn genug Daten da sind, auf die sich
etwas setzen kann** — bei 2 Mio. Schritten schadet dieselbe Abkühlung eher.
Deshalb ist `learning_rate_decay_steps` standardmäßig `None`, was „über den
ganzen Lauf" bedeutet: die Abkühlung folgt dem Budget statt nach einem
festen Bruchteil zu enden.

Die Score-Verteilung ist stark rechtsschief, ein Mittelwert allein führt in
die Irre. Perzentile eines Laufs mit dem *alten* 2-Mio-Budget:

| Perzentil | 10 | 25 | 50 | 75 | 90 | max |
|---|---|---|---|---|---|---|
| Score | 6 | 18 | 48 | 95 | 153 | 363 |

### Woran der Vogel stirbt

Über 120 greedy Episoden, gemessen im letzten Frame vor dem Aufprall:

| Ursache | Anteil |
|---|---|
| **obere Röhre** (Vogel zu hoch) | **0.64** |
| untere Röhre | 0.28 |
| Boden | 0.00 |

Mittleres `dy` im Todesframe: −40 px. Der Vogel schießt also überwiegend nach
oben über. Das passt zur Physik: ein Flügelschlag setzt die Geschwindigkeit
schlagartig auf −9 und der Vogel steigt danach etwa neun Frames lang um
insgesamt ~45 px, während nach unten nur die Gravitation mit +1 pro Frame
wirkt. Aufwärtskorrekturen sind grob, Abwärtskorrekturen fein. Wer die
Rasterisierung weiter verbessern will, sollte dort ansetzen.

## Welche Zahl wofür

| Zahl | Aussage | Grenzen |
|---|---|---|
| `mean_score` aus `evaluate.py` | was die fertige Policy kann | zensierbar durch das Frame-Limit, siehe `truncation_rate` |
| `steps_to_<n>` | Sample-Effizienz: Schritte, bis der gleitende Score das Niveau hält | nicht zensierbar, **einzige über Verfahren hinweg vergleichbare Zahl** |
| `mean_return` | nur *innerhalb* eines Reward-Schemas vergleichbar | zwischen Schemata bedeutungslos |
| `coverage` | Anteil je aktualisierter Tabelleneinträge | **irreführend**, mehr ist nicht besser — siehe unten |
| `mean_flap_rate` | Verhaltensmaß | Schweben erfordert theoretisch 1/19 ≈ 0.053 |
| `mean_gap_offset` | Verhaltensmaß: Pixel zur Lückenmitte, **positiv = unterhalb** | systematische Lage, nicht die Streuung |
| `mean_abs_gap_offset` | wie eng die Mitte gehalten wird | eine symmetrisch schwingende Policy mittelt sich im Vorzeichen weg |

### `coverage` richtig lesen — und wofür sie nicht taugt

Die naheliegende Lesart „mehr Abdeckung ist besser" ist hier **falsch**. Das
beste Schema hat 0.36, das drittbeste 0.62.

Der Grund: der Nenner ist eine Eigenschaft des Gitters, nicht des Lernens. Die
erreichbaren Zustände bilden eine dünne Mannigfaltigkeit im (dx, dy, vel)-Raum,
weil Position, Geschwindigkeit und Röhrenabstand über die Dynamik gekoppelt
sind. Solange der Vogel *in* der Röhre ist (`dx < 0`), bedeutet `|dy| > 38`
bereits den Aufprall — die Lücke ist 100 px hoch, der Vogel 24 px. Verfeinert
man das Gitter, wächst die Zellenzahl viel schneller als die Zahl der Zellen,
die diese Mannigfaltigkeit schneiden:

| Gitter | Zustände | besucht | Abdeckung | Median Besuche | Score |
|---|---|---|---|---|---|
| 16×16×20 | 5 120 | 5 608 | 0.55 | 88 | 42 |
| **24×32×20** | 15 360 | 11 125 | **0.36** | 41 | **183** |
| 32×40×20 | 25 600 | 17 947 | 0.35 | 29 | 44 |
| 40×56×20 | 44 800 | 27 444 | 0.31 | 19 | 40 |
| 56×72×20 | 80 640 | 45 218 | 0.28 | 11 | 33 |

Die Zustandszahl steigt um Faktor 15.8, die besuchten Einträge nur um 8.1 —
also `besucht ≈ Zustände^0.76`. Die Abdeckung **muss** beim Verfeinern fallen,
unabhängig von der Lernqualität.

Der direkte Vergleich macht es deutlich: `adaptive` besuchte 12 594 Einträge
bei 0.62 Abdeckung, `uniform` 11 125 bei 0.36. Fast gleich viele **echte**
Situationen, jede ungefähr gleich oft gesehen (53 gegen 41 Besuche im Median) —
`uniform` trifft darin nur feinere Unterscheidungen, und genau das ergibt die
dreifach bessere Policy. Der Abdeckungsunterschied sind im Wesentlichen die
zusätzlichen 10 000 Zellen, die die Physik nie erzeugt.

Dazu kommt ein prinzipielles Problem: die Abdeckung fällt über die Gitterreihe
**monoton**, die Leistung hat aber ein Maximum in der Mitte. Eine monotone
Kennzahl kann ein unimodales Optimum nicht lokalisieren. Die beiden Fehlerarten
trennt stattdessen das Paar aus Auflösung und **Besuchen je besuchtem
Eintrag**: zu grob heißt viele Besuche (88), aber tödlich verschiedene
Situationen in einer Zelle; zu fein heißt feine Unterscheidungen bei 11
Besuchen, aus denen nichts zu lernen ist.

Wofür die Zahl weiterhin taugt: ein plötzlicher Einbruch zwischen zwei sonst
ähnlichen Konfigurationen ist ein Hinweis, und sie kostet nichts. Als Maß für
„gut exploriert" taugt sie nicht — bei `adaptive` deckte das Training 99.9 %
aller Zustände ab, die eine greedy, eine ε-greedy und eine Zufallspolicy
zusammen je erreichen, und stand trotzdem bei 0.62.

### Wozu die beiden Offsets

Zwei Reward-Schemata können denselben Score erreichen und dabei völlig
verschieden fliegen — genau das ist der Gegenstand einer Studie über Reward-
*Design*, und der Score allein zeigt es nicht. Beispiel aus zwei Seeds:

| Reward | Score | Flap-Rate | Offset | \|Offset\| |
|---|---|---|---|---|
| `legacy` | 16.6 / 47.6 | 0.074 / 0.077 | −1.2 / −4.6 | 22.3 / 20.7 |
| `risk_averse` | 64.2 / 40.0 | 0.088 / 0.075 | −1.7 / −3.5 | 23.1 / 20.5 |
| `energy` | 16.5 / 17.7 | **0.061 / 0.070** | **+1.2 / +4.0** | 24.9 / 24.0 |

`energy` bestraft Flügelschläge — und ist als einziges Schema erkennbar
sparsamer und fliegt *unterhalb* der Mitte, wo es weniger flattern muss. Die
Scores streuen dabei so stark, dass sie zwischen den Schemata nichts hergeben;
Flap-Rate und Offset-Vorzeichen sind über beide Seeds konsistent. Auch das ist
nur eine Plausibilitätsprüfung, kein Ergebnis — dafür fehlt der Signifikanztest.

**Achtung beim Vergleich mit PPO:** `rl/evaluate.py` auf Branch `fb_drl_ppo`
berechnet ein gleichnamiges `_gap_offset` **anders** — gemessen ab der
*Oberkante* des Vogels (`obs[9]`) und in Bildschirmhöhen statt Pixeln. Das Env
selbst rechnet in `_gap_potential()` mit der Mitte, deshalb tut das die
gemeinsame Fassung hier auch. Die beiden Zahlenreihen unterscheiden sich um
konstant 12 px und den Faktor 512 und dürfen nicht ohne Umrechnung in eine
Tabelle.

## Vier Dinge, die leicht schiefgehen

1. **Die Röhren-Slots sind nach x sortiert, nicht nach Rolle.** Nach dem Punkten
   hält Slot 0 für ~2 Frames die bereits passierte Röhre. Die nächste ist der
   erste Slot, dessen *rechte* Kante noch vor dem Vogel liegt — dieselbe Regel,
   die das Env in `_gap_potential()` anwendet.
2. **Der Platzhalter `(288, 0, 512)`** für Röhren hinter dem Bildschirm hat eine
   rechnerische Lückenmitte von 256, außerhalb des echten Bereichs `[150, 220]`.
   Zu Episodenbeginn liegen zwei Platzhalter bei exakt demselben x wie die echte
   Röhre; `pipes_ahead` wirft sie explizit weg, statt sich auf die Stabilität von
   `sorted` zu verlassen.
3. **Zwei Frame-Limits.** Training 3 000 (sonst frisst eine Episode das
   Budget), Messung 50 000. Aus dem Default-Lauf: 55 % der Trainings-
   Evaluationen liefen ins Limit, gemeldet wurden 63.5 — ungezensiert sind es
   **158.6**. Gegen das Trainingslimit gemessen fallen alle guten Varianten auf
   dieselbe Zahl. Das Mess-Limit liegt über den 20 000 der DQN-Arbeit, weil
   dieser Agent sie überholt hat: bei Score 836 läuft eine Episode rund 30 000
   Frames. Ein Score ist nur mit einem gegen *dasselbe* Limit gemessenen
   vergleichbar; `truncation_rate` sagt, wann es beißt.
4. **`private_zone` ist im Feature-Modus wirkungslos.** Ohne LIDAR ist
   `in_private_zone` hart `False`, der Term (−0.5 in `legacy` und `additive`)
   feuert nie. Ein Unterschied zwischen zwei Schemata, die sich *nur* darin
   unterscheiden, kann hier also nicht existieren.

## Module

| Modul | Rolle |
|---|---|
| `config.py` | `QLearningConfig` + `add_config_arguments` |
| `discretize.py` | `pipes_ahead`, `Discretizer`, `DISCRETIZERS`, `build_discretizer` |
| `agent.py` | `QLearningAgent`: Tabelle, Zeitpläne, Update, Serialisierung |
| `train.py` | Trainingsschleife, CLI |
| `evaluate.py` | ungezensierte Messung, Zuschauen, `RUN_LOADER` für die gemeinsame Auswertung |
| `experiments.py` | Studienarten in `STUDIES`, Prozess-Pool, Wiederaufnahme |

Geteilt mit den anderen Verfahren, neu auf `main` angelegt:

| Modul | Rolle |
|---|---|
| `rl/rewards.py` | die Reward-Schemata (unverändert, nur benutzt) |
| `rl/geometry.py` | das Lesen des Beobachtungsvektors: `pipes_ahead`, `gap_offset`; `env_gap_offset` liest dasselbe aus dem Spielzustand, für jede Beobachtungsart |
| `rl/rollout.py` | `make_env`, `reward_config_for`, `rollout`, `summarize_rollout`, `EVAL_SEED` |
| `rl/runlog.py` | `CsvLogger`, `EVAL_LOG_FIELDS`, `steps_to_thresholds` |
| `rl/analysis/` | die Auswertungskette, siehe „Studien: vier Stufen" |

`rl/geometry.py` ist bewusst geteilt: die Rasterisierung und die Offset-Messung
müssen sich darüber einig sein, welche Röhre gerade die aktuelle ist — die eine
misst, was die andere optimiert. Der Test
`test_agrees_with_the_environments_shaping_potential` prüft Frame für Frame
gegen `_gap_potential()` des Env, dass beide dieselbe Geometrie meinen.

`rl/rollout.py` und `rl/runlog.py` sind torch-frei und `config`-ducktyped: gelesen
werden nur `use_lidar`, `normalize_obs`, `pipe_gap`, `reward_preset`,
`reward_overrides`, `gamma`, `max_episode_steps`. Ein `DQNConfig` passt also
unverändert hinein — die Dopplung, die `dqn/ARBEITSSTAND.md` §13 als offen notiert,
lässt sich damit beim Merge auflösen.

## Tests

```bash
pytest flappy_bird_gymnasium/tests/test_qlearning.py        # Agent, Rasterisierung
pytest flappy_bird_gymnasium/tests/test_qlearning_study.py  # Studien, Anbindung an rl/analysis
pytest flappy_bird_gymnasium/tests/test_analysis.py         # die Auswertungskette selbst
```

`test_qlearning.py` hat 57 Tests, Laufzeit unter einer Sekunde. Der wichtigste ist
`test_features_match_the_environment_internals`: er vergleicht die gesamte
Reduktion Frame für Frame gegen die Interna der Simulation und würde jeden
Vorzeichenfehler, jeden falschen Beobachtungsindex und jede Verwechslung von
Vogel-Oberkante und -Mitte fangen — Fehler, die nicht abstürzen, sondern still
einen schlechteren Agenten erzeugen.

## Studien: vier Stufen

Alles, was in einen Bericht soll, läuft durch diese Kette. Jede Stufe schreibt
Dateien, die die nächste liest, sodass eine Auswertung reproduzierbar ist statt
im Terminal zu verschwinden.

```bash
# 1. trainieren: eine Achse variieren, alles andere festhalten
python -m flappy_bird_gymnasium.qlearning.experiments reward --seeds 5
python -m flappy_bird_gymnasium.qlearning.experiments discretizer --seeds 5
python -m flappy_bird_gymnasium.qlearning.experiments granularity --seeds 5
python -m flappy_bird_gymnasium.qlearning.experiments ablation --seeds 5
python -m flappy_bird_gymnasium.qlearning.experiments params --seeds 5
python -m flappy_bird_gymnasium.qlearning.experiments sweep \
    --param epsilon_decay_steps --values 150000 300000 3000000 --seeds 5

# 2. messen: alle Läufe gepaart auf denselben Episoden-Seeds
python -m flappy_bird_gymnasium.rl.analysis.summarize runs/study_reward \
    --algorithm qlearning --episodes 50

# 3. testen: ist der Unterschied größer als das Rauschen zwischen Seeds?
python -m flappy_bird_gymnasium.rl.analysis.significance runs/study_reward \
    --baseline legacy

# 4. zeichnen und aufnehmen
python -m flappy_bird_gymnasium.rl.analysis.plot runs/study_reward --out docs/reward.png
python -m flappy_bird_gymnasium.rl.analysis.plot runs/study_reward --behaviour \
    --out docs/reward_verhalten.png
python -m flappy_bird_gymnasium.rl.analysis.plot runs/study_reward --runs \
    --panels score td_error coverage flap_rate --out docs/diag.png
python -m flappy_bird_gymnasium.rl.analysis.record runs/study_reward/legacy_seed0/best.npz \
    --algorithm qlearning --out docs/legacy.gif --seconds 20 --skip-to-pipe 5
```

Die Stufen 2–4 liegen in `rl/analysis/` und sind für alle Verfahren dieselben;
Q-Learning ist darin nur über `RUN_LOADER` in `evaluate.py` angebunden.

`--baseline` ist Pflicht und erwartet den Variantennamen, wie er im Verzeichnis
steht. Alphabetisch wäre fast nie die gemeinte Referenz:

| Studie | Referenz |
|---|---|
| `reward` | `legacy` (alphabetisch erste wäre `additive`) |
| `discretizer` | `uniform` (alphabetisch erste wäre `adaptive`) |
| `granularity` | `g15k` |
| `ablation` | `basis` |
| `params` | `baseline` |
| `seeds` | `default` |

Eine unterbrochene Studie lässt sich fortsetzen: derselbe Aufruf überspringt
jeden Lauf, dessen `config.json` **exakt** der angeforderten Konfiguration
entspricht. Weniger streng zu prüfen wäre gefährlich — eine Studie, die
klammheimlich Ergebnisse zweier Einstellungen unter einem Label mischt, sieht
völlig normal aus.

Jede Messung landet in einer eigenen Datei: `evaluations.csv` für `best.npz`,
`evaluations_latest.csv`, `evaluations_limit200000.csv`,
`evaluations_shaped+sparse.csv` für eine Teilmenge über `--variants`. Der Test
schreibt `significance_<messung>_vs_<referenz>.csv` daneben.

### Was die Module in `rl/analysis/` tun

| Modul | Rolle |
|---|---|
| `runs.py` | Vertrag eines Laufverzeichnisses, `RunLoader` je Verfahren, Registry `LOADERS` |
| `summarize.py` | gepaarte Messung, Zufallsreferenz, Aggregation je Variante, `--variants` zum Nachmessen |
| `significance.py` | Permutationstest auf der Mann-Whitney-U-Statistik |
| `plot.py` | Lernkurven mit Quartilband, Verhalten je Seed, Diagnose einzelner Läufe |
| `record.py` | GIF einer Greedy-Episode, mit `.txt` daneben, was es zeigt |

Vier Festlegungen, die von den gemessenen Eigenschaften dieser Daten kommen:

* **Median statt Mittelwert** als Kennzahl. Der Score ist stark rechtsschief;
  vier Seeds derselben Konfiguration ergaben Mittelwerte von 51, 60, 49 und 455.
* **Punkte je Seed statt Balken mit Fehlerbalken.** Bei fünf Läufen *sind* die
  einzelnen Punkte das Ergebnis, und ein Balken verbirgt, ob sie sich einig waren.
* **Exakter Permutationstest**, keine t-Statistik. Bei wenigen Seeds wird
  vollständig enumeriert — und das heißt auch, dass p eine Untergrenze hat:
  vier gegen vier Seeds kommen nie unter 0.029, sechs gegen sechs nie unter
  0.0022. Ein Vergleich, der „nicht signifikant" ist, kann allein an der
  Seed-Zahl scheitern. `significance.py` schreibt diese Grenze unter jede
  Tabelle.
* **„Nie erreicht" zählt als langsamster Rang**, nicht als fehlender Wert. Ein
  Seed ohne `steps_to_<n>` geht mit `inf` in den Rangtest ein. Wegwerfen hieße,
  eine Variante nur auf ihren Glücks-Seeds zu vergleichen.

## Stand der Zahlen in dieser Datei

Die Tabellen oben stammen aus Vorlauf-Messungen mit vier bis sechs Seeds, die
**nicht** durch diese Kette gelaufen sind. Sie begründen die Defaults; als
Ergebnis in einem Bericht taugen sie nicht. Für den Bericht gilt: Studie über
`experiments.py` fahren, mit `summarize.py` messen, mit `significance.py`
testen, mit `plot.py` zeichnen — dann liegt alles unter `runs/study_<name>/`
und ist nachvollziehbar.

Fertig gesetzte Abbildungen über mehrere Studien hinweg (wie `dqn/figures.py`
auf `fb_drl_dqn`) erzeugt `figures.py`, aufgebaut auf der geteilten Kette:

```bash
python -m flappy_bird_gymnasium.qlearning.figures <runs> --check   # Zahlen gegenpruefen
python -m flappy_bird_gymnasium.qlearning.figures <runs> --out docs/evaluation/praesentation
```

`<runs>` enthält `study_matrix`, `gamma_coupled` und `nstep`.
