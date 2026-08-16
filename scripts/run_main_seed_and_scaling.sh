#!/usr/bin/env bash
set -euo pipefail

# Runs the next main-seed and scaling experiments:
#   1. d128 L1 fan-recurrent seeds 1,2
#   2. basic direct L6/L10 seeds 1,2
#   3. d128 L2 fan-recurrent seed 0
#   4. d256 L2 fan-recurrent seed 0
#
# Override defaults from the shell, for example:
#   DEVICE=cuda bash scripts/run_main_seed_and_scaling.sh

DEVICE="${DEVICE:-mps}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
BOUNDARY_DATA_DIR="${BOUNDARY_DATA_DIR:-data/boundary_sweep}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/original}"

read -r -a FIXED_LOOP_COUNTS <<< "${FIXED_LOOP_COUNTS:-20 24 40 64 128}"

if [[ ! -f "${BOUNDARY_DATA_DIR}/boundary_11_19.jsonl" ]]; then
  echo "missing ${BOUNDARY_DATA_DIR}/boundary_11_19.jsonl"
  echo "generate it first with: python -m src.data.data --boundary-sweep --out ${BOUNDARY_DATA_DIR} --boundary-min-swaps 11 --boundary-max-swaps 19 --samples-per-length 100 --seed 123"
  exit 1
fi

latest_run_dir() {
  local run_name="$1"
  local run_dir
  run_dir=$(ls -dt "${OUTPUT_DIR}/${run_name}__"* 2>/dev/null | head -1 || true)
  if [[ -z "${run_dir}" ]]; then
    echo "could not find run directory for ${run_name}" >&2
    exit 1
  fi
  printf '%s\n' "${run_dir}"
}

eval_fan_run() {
  local run_dir="$1"

  python scripts/evaluate_length_matched_checkpoint.py \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --swaps-per-loop 1 \
    --splits id_test ood_x4 ood_x8 \
    --fixed-loop-counts "${FIXED_LOOP_COUNTS[@]}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --out "${run_dir}/length_matched_k_sweep.json"

  python scripts/evaluate_length_matched_checkpoint.py \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --data-dir "${BOUNDARY_DATA_DIR}" \
    --splits boundary_11_19 \
    --swaps-per-loop 1 \
    --fixed-loop-counts "${FIXED_LOOP_COUNTS[@]}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --out "${run_dir}/boundary_11_19_k_sweep.json"
}

run_fan() {
  local seed="$1"
  local d_model="$2"
  local n_heads="$3"
  local d_ff="$4"
  local num_layers="$5"
  local run_name="$6"

  python scripts/run_original_experiments.py \
    --architecture fan-recurrent \
    --position-encoding none \
    --fan-input-format atomic \
    --d-model "${d_model}" \
    --n-heads "${n_heads}" \
    --d-ff "${d_ff}" \
    --num-layers "${num_layers}" \
    --num-loops 10 \
    --dropout 0.0 \
    --online-training \
    --train-steps "${TRAIN_STEPS}" \
    --curriculum-min-swaps 2 \
    --curriculum-max-swaps 10 \
    --curriculum-steps-per-length 1000 \
    --swaps-per-loop 1 \
    --length-matched-eval \
    --deep-supervision-weight 0 \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --grad-clip "${GRAD_CLIP}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --run-name "${run_name}"

  eval_fan_run "$(latest_run_dir "${run_name}")"
}

run_basic() {
  local seed="$1"
  local num_layers="$2"
  local run_name="basic-atomic-nope-causal-l${num_layers}-seed${seed}"

  python scripts/run_original_experiments.py \
    --architecture direct \
    --position-encoding none \
    --direct-input-format atomic \
    --direct-causal \
    --d-model 128 \
    --n-heads 4 \
    --d-ff 512 \
    --num-layers "${num_layers}" \
    --dropout 0.0 \
    --online-training \
    --train-steps "${TRAIN_STEPS}" \
    --curriculum-min-swaps 2 \
    --curriculum-max-swaps 10 \
    --curriculum-steps-per-length 1000 \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --grad-clip "${GRAD_CLIP}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --run-name "${run_name}"

  local run_dir
  run_dir="$(latest_run_dir "${run_name}")"
  python scripts/evaluate_direct_checkpoint.py \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --data-dir "${BOUNDARY_DATA_DIR}" \
    --splits boundary_11_19 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --out "${run_dir}/boundary_11_19.json"
}

for seed in 1 2; do
  run_fan \
    "${seed}" 128 4 512 1 \
    "fan-atomic-nope-curr2to10-seed${seed}"
done

for seed in 1 2; do
  run_basic "${seed}" 6
  run_basic "${seed}" 10
done

run_fan \
  0 128 4 512 2 \
  "fan-atomic-nope-scaleD-d128-h4-ff512-L2-curr2to10-seed0"

run_fan \
  0 256 8 1024 2 \
  "fan-atomic-nope-scaleWD-d256-h8-ff1024-L2-curr2to10-seed0"
