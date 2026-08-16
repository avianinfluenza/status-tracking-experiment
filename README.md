# status-tracking-experiment

YAI 2026 여름방학 기초NLP연구팀 Toy project

자연어로 서술된 상태추적 과제에서 recurrent computation이 실제 추가 계산
자원으로 사용되는지를 실험적으로 검증한다. 현재 레포에는 두 실험 축이 함께
있다.

- `src/original/`: 현재 주 실험인 5인 ball-swap. Direct, Explicit CoT, 기존 Recurrent, R0, Fan-aligned Looped Transformer, event-wise recurrence를 비교한다.
- `src/systematic/`: target transition depth, distractor, total events를 독립 통제하는 object-location controlled track. 이후 보조 실험으로 실행한다.

주 실험의 가설, 비교 조건, 해석 기준은 [RESEARCH_DESIGN.md](RESEARCH_DESIGN.md)와
[MODELING.md](MODELING.md)를 참고한다. 기존 ball-swap 데이터 형식은
[docs/DATASET.md](docs/DATASET.md)에 정리되어 있다.

## 구조

```text
data/               train / id_test / ood_x4 / ood_x8 JSONL
configs/            original/basic/looped/R0 및 systematic 설정
scripts/            단일·다중 seed 실행, systematic 집계와 그래프
src/data/           ball-swap vocabulary, collate, dataset, 데이터 검증
src/model/          Basic/Looped 진입점과 shared classifier
src/original/       Direct, Explicit CoT, Recurrent 및 학습·평가 구현
src/systematic/     controlled object-location/systematic 실험 파이프라인
src/trainer.py      YAML config를 original 실험 실행기로 연결
tests/              original 및 systematic 테스트
```

## 설치와 검증

```bash
python -m pip install -e '.[dev]'
python -m src.data.verify
pytest
```

## Original Ball-Swap

단일 모델의 빠른 실행 확인:

```bash
python scripts/run_original_experiments.py --architecture direct --smoke --device cpu
python scripts/run_original_experiments.py --architecture cot --smoke --device cpu
python scripts/run_original_experiments.py --architecture recurrent --smoke --device cpu \
  --adaptive-kl-eval
python scripts/run_original_experiments.py --architecture recurrent-r0 --smoke --device cpu \
  --eval-loop-counts 1 2 \
  --adaptive-kl-eval --adaptive-min-confidence 0.0 --adaptive-update-threshold 1000000000
python scripts/run_original_experiments.py --architecture event-recurrent --smoke --device cpu
python scripts/run_original_experiments.py --architecture fan-recurrent \
  --position-encoding none --online-training --smoke --device cpu

# Fixed output registers at the start of the sequence (accepts True/False too)
python scripts/run_original_experiments.py --architecture recurrent-r0 \
  --slot_first True --num-loops 6 --seed 0 --device cuda

# Extended-length profile: train/ID 2~32, OOD x4 40~80, OOD x8 80~160
python scripts/run_original_experiments.py --architecture recurrent-r0 \
  --extended-length --slot_first True --num-loops 6 --seed 0 --device cuda

# Evaluation-only loop-by-loop trajectory probe
python scripts/run_original_experiments.py --architecture recurrent-r0 \
  --trajectory-probe-eval --eval-loop-counts 2 4 6 12 24 \
  --seed 0 --device cuda
```

The extended-length JSONL files are stored under `data/extended_length/`. To
regenerate them deterministically:

```bash
python -m src.data.data --extended-length --out data/extended_length \
  --n-train 10000 --n-test 500 --seed 0
```

학습 경계 직후인 11~20 swaps를 길이별 100개씩 평가하려면:

```bash
python -m src.data.data --boundary-sweep --out data/boundary_sweep \
  --boundary-min-swaps 11 --boundary-max-swaps 20 \
  --samples-per-length 100 --seed 0

python scripts/evaluate_length_matched_checkpoint.py \
  --checkpoint runs/original/<RUN>/checkpoint.pt \
  --data-dir data/boundary_sweep --splits boundary_11_20 \
  --swaps-per-loop 1 --fixed-loop-counts 8 10 12 16 20 \
  --device mps --out runs/original/<RUN>/boundary_11_20.json
```

출력 SLOT의 absolute-position OOD만 분리하는 2단계 ablation은 기존
random-loop baseline과 모든 설정을 같게 두고 `--slot-first True`만 추가한다.
실행 이름에는 자동으로 `slotfirst`가 기록된다. checkpoint-only 평가기는 같은
디렉터리의 `args.json`에서 이 값을 자동으로 복원하며, 필요할 때만
`--slot-first` 또는 `--no-slot-first`로 덮어쓴다.

3단계 `event-recurrent` 모델은 토큰열 밖의 `[B, 5, D]` latent state register를
사용하고, 각 recurrent step에서 현재 swap 문장 7토큰만 shared update에
제공한다. register와 event에는 global event/sequence position이나 loop
embedding을 주지 않는다. state는 step 사이에 그대로 전달된다.

`event-recurrent`의 학습 목적함수는 최종 상태 CE뿐이다.
`--event-trajectory-probe`와 no-op 입력 평가는 진단 전용이며 학습 loss에 관여하지
않는다.

```bash
python scripts/run_original_experiments.py \
  --architecture event-recurrent \
  --event-trajectory-probe \
  --epochs 100 --seed 0 --device mps
```

학습 후 boundary split은 `scripts/evaluate_event_checkpoint.py`로 평가한다.

YAML config 진입점:

```bash
python main.py --config configs/basic_model.yaml --smoke --device cpu
python main.py --config configs/looped_model.yaml --smoke --device cpu
python main.py --config configs/recurrent_r0_model.yaml --smoke --device cpu
python main.py --config configs/fan_recurrent_model.yaml --smoke --device cpu
```

R0 ball-swap 주 조건은 embedding reinjection이 없는 `h <- shared_block(h)`이다.
학습 loop보다 큰 inference loop sweep은 evaluation에만 적용한다.
`--deep-supervision-weight`는 다른 기여자가 만든 기존 ablation으로 보존한다.
값이 0보다 크면 non-final loop에도 최종 상태 CE를 적용한다. 기본값 0에서는
최종 loop의 CE만 학습한다.

`--swaps-per-loop r`는 random loop sampling 대신 샘플별로
`ceil(n_swaps / r)`회 recurrence를 사용한다. 같은 계산 예산을 가진 샘플만 한
학습 배치에 묶는다. 이 옵션은 중간 상태를 감독하지 않으며 마지막 loop 출력에만
최종 상태 CE를 건다.
평가의 `--adaptive-max-loops`는 `--adaptive-kl-eval`과 함께만 쓰며 swap 수를
입력으로 받지 않는 halt signal의 상한이다.
`--length-matched-eval`은 데이터의 swap 수로 각 샘플을 `ceil(n_swaps / r)`회
실행하는 **진단용 oracle**이다. 결과의 `length_matched_oracle`만 이 규칙을
사용하며, adaptive computation 성능으로 보고하면 안 된다.

```bash
python scripts/run_original_experiments.py \
  --architecture recurrent-r0 \
  --position-encoding sinusoidal \
  --num-loops 6 \
  --epochs 30 \
  --seed 0 \
  --device cuda \
  --eval-loop-counts 1 2 4 6 8 12 16 24
```

### Fan-aligned condition

`fan-recurrent`는 Looped Transformers for Length Generalization과 방법론을
맞추기 위한 별도 조건이다.

- NoPE + causal attention
- `h_0 = 0`, `h_k = F_theta(h_{k-1} + embed(x))`
- 모든 loop에서 동일한 `num_layers` depth stack 재사용
- 샘플의 swap 수 `N`에 대해 기본 `K=N` (`--swaps-per-loop 1`)
- 중간 상태 loss 없이 마지막 loop의 최종 상태 CE만 사용
- 2→10 swaps curriculum, fresh online batches, curriculum 이후 cosine LR decay

온라인 데이터는 전역 RNG에 의존하지 않는다. 각 예제는
`(seed, optimizer_step, sample_index)`에서 결정적으로 생성되므로 같은 명령과
시드는 모든 학습 배치를 동일하게 재현한다. 실행 설정과 seed scheme은
`args.json`과 `result.json.training_regime`에 기록된다. 온라인 배치에서는
`intermediate_states`를 제거해 trajectory label이 학습 코드에 들어갈 수 없게
한다.

### Fan diagnostic controls

두 조건은 주 Fan 결과와 섞어 보고하지 않는 진단용 control이다. 둘 다 동일한
causal recurrent block, input injection, online curriculum, final-state CE를
사용한다.

- **template + sinusoidal**: 기존 한국어 template를 유지하되 순서 정보를
  제공하는 control이다. `--fan-positional-control`을 명시해야만
  `fan-recurrent`에서 sinusoidal encoding을 사용할 수 있으며 결과 track은
  `fan_template_sinusoidal_control`로 기록된다.
- **atomic + NoPE**: `[INIT_person_color]` 하나와 `[SWAP_left_right]` 하나가
  각각 한 token이 되는 구조화 입력이다. 이 조건은 NoPE를 유지하고 결과 track을
  `fan_atomic_nope_control`로 기록한다. 자연어 template의 parsing 비용을 제거하지만
  task와 최종 five-colour labels는 동일하다.

```bash
# Template + sinusoidal positional control (not the Fan main condition).
python scripts/run_original_experiments.py \
  --architecture fan-recurrent \
  --position-encoding sinusoidal \
  --fan-positional-control \
  --online-training \
  --train-steps 100000 \
  --curriculum-min-swaps 2 --curriculum-max-swaps 10 \
  --curriculum-steps-per-length 1000 \
  --swaps-per-loop 1 \
  --deep-supervision-weight 0 \
  --seed 0 --device mps

# Atomic semantic tokens + NoPE control.
python scripts/run_original_experiments.py \
  --architecture fan-recurrent \
  --position-encoding none \
  --fan-input-format atomic \
  --online-training \
  --train-steps 100000 \
  --curriculum-min-swaps 2 --curriculum-max-swaps 10 \
  --curriculum-steps-per-length 1000 \
  --swaps-per-loop 1 \
  --deep-supervision-weight 0 \
  --seed 0 --device mps
```

단일 seed 주 실험:

```bash
python scripts/run_original_experiments.py \
  --architecture fan-recurrent \
  --position-encoding none \
  --num-layers 1 \
  --num-loops 10 \
  --online-training \
  --train-steps 100000 \
  --curriculum-min-swaps 2 \
  --curriculum-max-swaps 10 \
  --curriculum-steps-per-length 1000 \
  --swaps-per-loop 1 \
  --batch-size 128 \
  --lr 0.0003 \
  --weight-decay 0.01 \
  --seed 0 \
  --device mps
```

5개 seed 재현:

```bash
python scripts/run_original_multiseed.py \
  --architectures fan-recurrent \
  --position-encoding none \
  --num-layers 1 \
  --num-loops 10 \
  --online-training \
  --train-steps 100000 \
  --curriculum-min-swaps 2 \
  --curriculum-max-swaps 10 \
  --curriculum-steps-per-length 1000 \
  --swaps-per-loop 1 \
  --seeds 0 1 2 3 4 \
  --device mps
```

주 결과의 `id_test`, `ood_x4`, `ood_x8`는 각각 sample별 `K=N`으로 자동
평가된다. `--eval-loop-counts`는 고정 compute 진단을 추가할 때만 사용한다.
중간 상태 정확도가 필요하면 `--trajectory-probe-eval`을 추가할 수 있지만 이는
평가 전용이다.

Advanced ablation은 주 조건과 섞지 않고 별도 run으로 실행한다.

```bash
# loop identity ablation
python scripts/run_original_experiments.py --architecture recurrent-r0 --num-loops 6 \
  --loop-conditioning learned --seed 0 --device cuda \
  --eval-loop-counts 1 2 4 6 8 12 16 24

# residual scaling ablation
python scripts/run_original_experiments.py --architecture recurrent-r0 --num-loops 6 \
  --residual-scale 0.5 --seed 0 --device cuda \
  --eval-loop-counts 1 2 4 6 8 12 16 24

# randomized loop-count training ablation
python scripts/run_original_experiments.py --architecture recurrent-r0 --num-loops 6 \
  --random-loops --random-min-loops 2 --random-max-loops 6 \
  --seed 0 --device cuda \
  --eval-loop-counts 1 2 4 6 8 12 16 24

# strengthened adaptive halting evaluation
python scripts/run_original_experiments.py --architecture recurrent-r0 --num-loops 6 \
  --adaptive-kl-eval --kl-threshold 0.001 \
  --adaptive-min-confidence 0.7 --adaptive-update-threshold 0.05 \
  --halting-patience 2 --seed 0 --device cuda
```

3개 seed의 Basic/Looped 비교와 swap-length별 집계:

```bash
python scripts/run_original_multiseed.py \
  --seeds 0 1 2 \
  --architectures direct recurrent-r0 \
  --position-encoding sinusoidal \
  --eval-loop-counts 1 2 4 6 8 12 16 24

python scripts/plot_original_results.py \
  runs/original/<multiseed-suite>/aggregate/summary.csv \
  --output-dir runs/original/figures
```

각 단일 run은 `runs/original/<run_name>__YYYYMMDD-HHMMSS[-NN]/` 폴더를 새로
만든다. 같은 모델을 반복 실행해도 기존 결과를 덮어쓰지 않는다. 폴더 안에는
`config.json`, `args.json`, `result.json`, `checkpoint.pt`가 저장된다.
`result.json`에는 ID/OOD별 slot accuracy, 5-person exact match, swap 횟수별
exact match, `started_at`, `finished_at`, `total_seconds`, 각 저장 경로가 기록된다.
Adaptive halting을 사용하면 평균 loop 수, halt rate, 마지막 symmetric KL, R0의
confidence/update ratio도 함께 저장한다.

`run_original_multiseed.py`는 `runs/original/multiseed-...__YYYYMMDD-HHMMSS/`
suite 폴더를 만들고, seed별 run 폴더와 `aggregate/`, `manifest.json`을 그 안에
함께 둔다.

## Systematic Controlled Experiments

기존 ball-swap은 전체 event 수와 target transition depth가 섞일 수 있으므로,
나중에 이 통제 실험을 실행할 때는 `scripts/run_systematic_experiments.py`를 사용한다. 이 경로는
symbolic simulator가 gold를 만들고 `target_depth`, distractor, total events를
각각 기록한다.

R0의 기본값은 `loop_conditioning=none`, `residual_scale=1.0`이며 학습 loop보다
큰 inference loop를 허용한다. Conditioning, random-loop training, residual
scaling, adaptive halting은 주 조건과 섞지 않고 각각 ablation으로 보고한다.

```bash
python scripts/run_systematic_experiments.py --smoke
python scripts/run_systematic_experiments.py --architecture standard --num-layers 6 --seed 0
python scripts/run_systematic_experiments.py --architecture recurrent --train-loops 6 --seed 0
```

여러 seed 실행 후 다음 명령으로 raw long-format CSV와 통계를 만든다.

```bash
python scripts/aggregate_systematic_results.py runs/systematic/*.json \
  --out-dir runs/systematic/aggregate

python -m pip install -e '.[analysis]'
python scripts/plot_systematic_results.py runs/systematic/aggregate/summary.csv \
  --out-dir runs/systematic/figures
```

`raw_long.csv`는 `model × seed × depth × loop × condition` 단위를 유지한다. JSON에는
optimizer, scheduler, generator version, checkpoint hash, parameter 수, FLOPs 추정,
latency, peak memory, gradient/hidden norm이 기록된다.
