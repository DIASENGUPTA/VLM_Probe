# #!/bin/bash
# # =============================================================================
# # generate_scenes.sh  --  launch probe-free scene generation.
# #   for each benchmark (baseline) x each adjective x N fixed seeds.
# # The model is loaded ONCE inside generate_scenes.py; this script just wires args.
# #
# # Usage:
# #   ./generate_scenes.sh <benchmark_glob> <save_root> "<adj1,adj2,...>" [num_seeds]
# #
# # Example (5 adjectives, 50 seeds each, same seeds for every condition):
# #   ./generate_scenes.sh '/home/kathakoli/VLM_probe/benchmark/*/*.json' \
# #       gen_scenes "messy,cluttered,minimalist,cozy,spacious" 50
# #
# # Env knobs (all optional):
# #   GEN_PY=python                 python interpreter (use your sage env's python)
# #   GEN_SCRIPT=./generate_scenes.py
# #   MODEL_PATH=Qwen/Qwen3.5-27B
# #   DECODE=sample                 sample (needed for seed diversity) | greedy
# #   TEMPERATURE=0.5
# #   MAX_TOKENS=8192
# #   BASE_SEED=1234
# #   SEED_STRIDE=10000
# #   ADJ_TEMPLATE=" The room is {adj}."   how the adjective joins the env note (must contain {adj})
# #   INCLUDE_BASELINE=0            1 -> also generate a no-adjective 'baseline' condition
# #   RESET_SEED_PER_SCENE=0        1 -> per-scene RNG isolation (else faithful to probe_backend)
# #   MAX_BENCHMARKS=               cap number of benchmarks (debug)
# #   NO_RESUME=0                   1 -> regenerate even if a scene already exists
# # =============================================================================

# BENCH_GLOB="$1"
# SAVE_ROOT="$2"
# ADJECTIVES="$3"
# NUM_SEEDS="${4:-50}"

# if [[ -z "$BENCH_GLOB" || -z "$SAVE_ROOT" || -z "$ADJECTIVES" ]]; then
#     echo "Usage: $0 <benchmark_glob> <save_root> \"<adj1,adj2,...>\" [num_seeds]"
#     echo "  e.g. $0 'benchmark/*/*.json' gen_scenes \"messy,cluttered,minimalist,cozy,spacious\" 50"
#     exit 1
# fi

# GEN_PY="${GEN_PY:-python}"
# GEN_SCRIPT="${GEN_SCRIPT:-./generate_scenes.py}"
# MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-27B}"
# DECODE="${DECODE:-sample}"
# TEMPERATURE="${TEMPERATURE:-0.5}"
# MAX_TOKENS="${MAX_TOKENS:-8192}"
# BASE_SEED="${BASE_SEED:-1234}"
# SEED_STRIDE="${SEED_STRIDE:-10000}"
# # NOTE: literal assignment (not ${VAR:-default}) because the default contains
# # '{adj}', whose '}' would prematurely close a ${...:-...} expansion.
# if [[ -z "${ADJ_TEMPLATE+set}" ]]; then
#     ADJ_TEMPLATE=' The room is {adj}.'
# fi
# INCLUDE_BASELINE="${INCLUDE_BASELINE:-0}"
# RESET_SEED_PER_SCENE="${RESET_SEED_PER_SCENE:-0}"
# NO_RESUME="${NO_RESUME:-0}"

# ARGS=(
#     --benchmark-glob "$BENCH_GLOB"
#     --save-root      "$SAVE_ROOT"
#     --adjectives     "$ADJECTIVES"
#     --adj-template   "$ADJ_TEMPLATE"
#     --num-seeds      "$NUM_SEEDS"
#     --base-seed      "$BASE_SEED"
#     --seed-stride    "$SEED_STRIDE"
#     --decode         "$DECODE"
#     --temperature    "$TEMPERATURE"
#     --max-tokens     "$MAX_TOKENS"
#     --model-path     "$MODEL_PATH"
# )
# [[ "$INCLUDE_BASELINE" == "1" ]]     && ARGS+=(--include-baseline)
# [[ "$RESET_SEED_PER_SCENE" == "1" ]] && ARGS+=(--reset-seed-per-scene)
# [[ "$NO_RESUME" == "1" ]]            && ARGS+=(--no-resume)
# [[ -n "$MAX_BENCHMARKS" ]]           && ARGS+=(--max-benchmarks "$MAX_BENCHMARKS")

# echo "=== scene generation ==="
# echo "glob:       $BENCH_GLOB"
# echo "save_root:  $SAVE_ROOT"
# echo "adjectives: $ADJECTIVES"
# echo "num_seeds:  $NUM_SEEDS   (base=$BASE_SEED stride=$SEED_STRIDE, same seeds for every adjective)"
# echo "decode:     $DECODE   template: '$ADJ_TEMPLATE'"
# echo "------------------------------------"

# exec "$GEN_PY" "$GEN_SCRIPT" "${ARGS[@]}"

#!/bin/bash
# =============================================================================
# generate_scenes.sh  --  launch probe-free scene generation.
#   for each benchmark (baseline) x each adjective x N fixed seeds.
# The model is loaded ONCE inside generate_scenes.py; this script just wires args.
#
# Usage:
#   ./generate_scenes.sh <benchmark_glob> <save_root> "<adj1,adj2,...>" [num_seeds]
#
# Example (5 adjectives, 50 seeds each, same seeds for every condition):
#   ./generate_scenes.sh '/home/kathakoli/VLM_probe/benchmark/*/*.json' \
#       gen_scenes "messy,cluttered,minimalist,cozy,spacious" 50
#
# Env knobs (all optional):
#   GEN_PY=python                 python interpreter (use your sage env's python)
#   GEN_SCRIPT=./generate_scenes.py
#   MODEL_PATH=Qwen/Qwen3.5-27B
#   DECODE=sample                 sample (needed for seed diversity) | greedy
#   TEMPERATURE=0.5
#   MAX_TOKENS=8192
#   BASE_SEED=1234
#   SEED_STRIDE=10000
#   ADJ_TEMPLATE=" The room is {adj}."   how the adjective joins the env note (must contain {adj})
#   INCLUDE_BASELINE=0            1 -> also generate a no-adjective 'baseline' condition
#   RESET_SEED_PER_SCENE=0        1 -> per-scene RNG isolation (else faithful to probe_backend)
#   PLACEMENT_N=5                 best-of-N placement candidates per scene; 1 = single-shot
#                                 (no feedback loop). 5 = faithful to the probed runs.
#   MAX_BENCHMARKS=               cap number of benchmarks (debug)
#   NO_RESUME=0                   1 -> regenerate even if a scene already exists
# =============================================================================

BENCH_GLOB="$1"
SAVE_ROOT="$2"
ADJECTIVES="$3"
NUM_SEEDS="${4:-50}"

if [[ -z "$BENCH_GLOB" || -z "$SAVE_ROOT" || -z "$ADJECTIVES" ]]; then
    echo "Usage: $0 <benchmark_glob> <save_root> \"<adj1,adj2,...>\" [num_seeds]"
    echo "  e.g. $0 'benchmark/*/*.json' gen_scenes \"messy,cluttered,minimalist,cozy,spacious\" 50"
    exit 1
fi

GEN_PY="${GEN_PY:-python}"
GEN_SCRIPT="${GEN_SCRIPT:-./generate_scenes.py}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-27B}"
DECODE="${DECODE:-sample}"
TEMPERATURE="${TEMPERATURE:-0.5}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
BASE_SEED="${BASE_SEED:-1234}"
SEED_STRIDE="${SEED_STRIDE:-10000}"
# NOTE: literal assignment (not ${VAR:-default}) because the default contains
# '{adj}', whose '}' would prematurely close a ${...:-...} expansion.
if [[ -z "${ADJ_TEMPLATE+set}" ]]; then
    ADJ_TEMPLATE=' The room is {adj}.'
fi
INCLUDE_BASELINE="${INCLUDE_BASELINE:-0}"
RESET_SEED_PER_SCENE="${RESET_SEED_PER_SCENE:-0}"
PLACEMENT_N="${PLACEMENT_N:-5}"
NO_RESUME="${NO_RESUME:-0}"

ARGS=(
    --benchmark-glob "$BENCH_GLOB"
    --save-root      "$SAVE_ROOT"
    --adjectives     "$ADJECTIVES"
    --adj-template   "$ADJ_TEMPLATE"
    --num-seeds      "$NUM_SEEDS"
    --base-seed      "$BASE_SEED"
    --seed-stride    "$SEED_STRIDE"
    --decode         "$DECODE"
    --temperature    "$TEMPERATURE"
    --max-tokens     "$MAX_TOKENS"
    --model-path     "$MODEL_PATH"
    --placement-n    "$PLACEMENT_N"
)
[[ "$INCLUDE_BASELINE" == "1" ]]     && ARGS+=(--include-baseline)
[[ "$RESET_SEED_PER_SCENE" == "1" ]] && ARGS+=(--reset-seed-per-scene)
[[ "$NO_RESUME" == "1" ]]            && ARGS+=(--no-resume)
[[ -n "$MAX_BENCHMARKS" ]]           && ARGS+=(--max-benchmarks "$MAX_BENCHMARKS")

echo "=== scene generation ==="
echo "glob:       $BENCH_GLOB"
echo "save_root:  $SAVE_ROOT"
echo "adjectives: $ADJECTIVES"
echo "num_seeds:  $NUM_SEEDS   (base=$BASE_SEED stride=$SEED_STRIDE, same seeds for every adjective)"
echo "decode:     $DECODE   template: '$ADJ_TEMPLATE'"
echo "------------------------------------"

exec "$GEN_PY" "$GEN_SCRIPT" "${ARGS[@]}"