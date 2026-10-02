#!/usr/bin/env bash
# Trains the CNN agent for every reward preset and seed, N runs in parallel.
#
#   scripts/run_all.sh                      # all presets, seeds 0 1 2, 2 parallel
#   REWARDS="legacy shaped" SEEDS="0" scripts/run_all.sh
#   PARALLEL=3 STEPS=500000 STUDY=cnn_v3 scripts/run_all.sh
#
# Runs go to runs/$STUDY/<reward>_seed<seed>/, the layout of the shared
# evaluation chain; afterwards
#   python -m flappy_bird_gymnasium.rl.analysis.summarize runs/$STUDY --algorithm cnn
# measures all of them on the same episodes (see README, "CNN agent").
#
# EXTRA holds the flags every run gets. For a new, comparable set of runs:
#   EXTRA="--explore-flap-prob 0.08 --eps-start 0.3 --eps-decay-steps 200000 \
#          --n-step 3 --buffer-size 250000" scripts/run_all.sh
#
# v3 candidate - the v2 set plus the network/optimiser stabilisers
# (LayerNorm, orthogonal init, Atari Adam epsilon), each also usable alone
# for an ablation:
#   EXTRA="--explore-flap-prob 0.08 --eps-start 0.3 --eps-decay-steps 200000 \
#          --n-step 3 --buffer-size 250000 \
#          --layer-norm --orthogonal-init --adam-eps 1.5e-4" scripts/run_all.sh
#
# Runs are ordered seed-major (all presets for seed 0, then seed 1, ...), so a
# partially finished queue still gives a complete comparison across rewards.
# A run whose directory already contains summary.json is skipped, so the script
# can be restarted after an interruption. The console output of each run goes
# to runs/$STUDY/<reward>_seed<seed>.log.
#
# On a Mac, 2 parallel runs give ~50% more total throughput than one, a third
# adds almost nothing (measured on an M5: 600 -> 900 -> 940 steps/s in total).

set -euo pipefail
cd "$(dirname "$0")/.."

REWARDS=${REWARDS:-"legacy additive sparse survival shaped risk_averse energy"}
SEEDS=${SEEDS:-"0 1 2"}
PARALLEL=${PARALLEL:-2}
STEPS=${STEPS:-1000000}
STUDY=${STUDY:-cnn}
EXTRA=${EXTRA:-""}
PYTHON=${PYTHON:-python}

mkdir -p "runs/$STUDY"

run_one() {
    local reward=$1 seed=$2
    local out="runs/${STUDY}/${reward}_seed${seed}"
    if [[ -f "$out/summary.json" ]]; then
        echo "skip  $out (summary.json exists)"
        return 0
    fi
    echo "start $out"
    # shellcheck disable=SC2086
    $PYTHON -m flappy_bird_gymnasium.train_cnn \
        --reward "$reward" --seed "$seed" --steps "$STEPS" --out "$out" \
        --no-export $EXTRA > "$out.log" 2>&1
    echo "done  $out: $(tail -n 1 "$out.log")"
}
export -f run_one
export PYTHON STEPS STUDY EXTRA

jobs=()
for seed in $SEEDS; do
    for reward in $REWARDS; do
        jobs+=("$reward $seed")
    done
done

printf '%s\n' "${jobs[@]}" | xargs -P "$PARALLEL" -L 1 bash -c 'run_one $0 $1'
echo "all runs finished"
