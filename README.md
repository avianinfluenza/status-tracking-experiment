# status-tracking-experiment

YAI 2026 여름방학 기초NLP연구팀 Toy project

자연어로 서술된 상태추적 과제에서 recurrent computation이 실제 추가 계산
자원으로 사용되는지를 실험적으로 검증한다. 현재 레포에는 두 실험 축이 함께
있다.

- `src/original/`: 현재 주 실험인 5인 ball-swap. Direct, Explicit CoT, 기존 Recurrent, R0 Recurrent를 비교한다.
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

# Fixed output registers at the start of the sequence (accepts True/False too)
python scripts/run_original_experiments.py --architecture recurrent-r0 \
  --slot_first True --num-loops 6 --seed 0 --device cuda

# Extended-length profile: train/ID 2~32, OOD x4 40~80, OOD x8 80~160
python scripts/run_original_experiments.py --architecture recurrent-r0 \
  --extended-length --slot_first True --num-loops 6 --seed 0 --device cuda
```

The extended-length JSONL files are stored under `data/extended_length/`. To
regenerate them deterministically:

```bash
python -m src.data.data --extended-length --out data/extended_length \
  --n-train 10000 --n-test 500 --seed 0
```

YAML config 진입점:

```bash
python main.py --config configs/basic_model.yaml --smoke --device cpu
python main.py --config configs/looped_model.yaml --smoke --device cpu
python main.py --config configs/recurrent_r0_model.yaml --smoke --device cpu
```

R0 ball-swap 주 조건은 embedding reinjection이 없는 `h <- shared_block(h)`이다.
학습 loop보다 큰 inference loop sweep은 evaluation에만 적용한다.

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
