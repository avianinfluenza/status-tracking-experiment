# 학습 인프라 업데이트 정리

## 개요

기존 학습 코드에서 부족했던 다음 다섯 가지 항목을 로컬 저장소에 구현했다.

1. 학습 데이터 기반 validation 분할
2. checkpoint 저장 및 중단된 학습 재개
3. epoch별 영구 로그 저장
4. YAML 설정과 실제 학습 코드 연결
5. AMP, scheduler, DataLoader 성능 옵션 지원

Git commit과 push는 수행하지 않았으며, 모든 변경 사항은 로컬 working tree에만 있다.
변경 전 기준 commit은 `cfb5bc5`이다.

## 1. Validation 분할

`train.jsonl` 내부 데이터를 train/validation으로 결정론적으로 분할한다. 현재는
단순 무작위 분할이 아니라 `n_swaps`별 층화(stratified) 분할을 사용한다.

- 기본 validation 비율: `0.1`
- 동일한 seed에서는 항상 동일한 분할 사용
- 각 swap length가 가능한 한 같은 비율로 train과 validation에 포함됨
- `id_test`, `ood_x4`, `ood_x8`은 학습 중 모델 선택에 사용하지 않음
- validation loss가 가장 낮은 epoch를 best checkpoint로 선정
- 최종 ID/OOD 평가는 `best.pt`를 불러온 후 수행

YAML 설정:

```yaml
validation:
  ratio: 0.1
```

직접 실행 옵션:

```bash
--validation-ratio 0.1
```

## 2. Checkpoint 저장 및 Resume

매 epoch가 끝날 때 완전한 학습 상태를 `last.pt`에 저장한다. Validation loss가
개선되면 `best.pt`도 갱신하고, 설정한 주기마다 `epoch_N.pt`를 추가로 남긴다.

Checkpoint에 포함되는 항목:

- 모델 가중치
- optimizer 상태
- scheduler 상태
- AMP GradScaler 상태
- 완료된 epoch
- 전체 epoch history
- best validation loss와 best epoch
- 누적 학습 시간
- Python, PyTorch, CUDA RNG 상태
- 모델 설정과 실제 실행 설정

저장 구조:

```text
runs/original/<run-name>/
├── metrics.jsonl
└── checkpoints/
    ├── best.pt
    ├── last.pt
    ├── epoch_5.pt
    ├── epoch_10.pt
    └── ...
```

최종 결과 파일은 기존 형식과의 호환성을 위해 다음 위치에도 저장한다.

```text
runs/original/<run-name>.json
runs/original/<run-name>.pt
```

YAML 설정:

```yaml
checkpointing:
  save_every: 5
  resume: null
  overwrite: false
```

중단된 동일 run을 자동으로 재개하려면 `resume`을 `auto`로 바꾼다.

```yaml
checkpointing:
  save_every: 5
  resume: auto
```

이때 `training.epochs`는 추가로 학습할 epoch 수가 아니라 최종 목표 epoch 수다.
예를 들어 epoch 12에서 중단된 30 epoch 실험은 `epochs: 30`으로 두고 재개한다.

직접 실행 예시:

```bash
python scripts/run_original_experiments.py \
  --architecture direct \
  --epochs 30 \
  --resume auto
```

특정 checkpoint를 지정할 수도 있다.

```bash
python scripts/run_original_experiments.py \
  --architecture direct \
  --epochs 30 \
  --resume runs/original/direct-sinusoidal-seed0/checkpoints/last.pt
```

모델 구조 설정이 checkpoint와 다르면 잘못된 재개를 막기 위해 오류를 발생시킨다.

### 동일 run 덮어쓰기 방지

같은 이름의 run 디렉터리, JSON 또는 PT 파일이 이미 존재하면 새 학습은 기본적으로
`FileExistsError`와 함께 중단된다. 다음 중 하나를 명시적으로 선택해야 한다.

- 기존 실험 계속하기: `resume: auto`
- 기존 실험을 삭제하고 새로 시작하기: `overwrite: true`
- 기존 결과 보존하기: 새로운 `run_name` 사용

```yaml
checkpointing:
  save_every: 5
  resume: null
  overwrite: true
```

`resume`과 `overwrite`는 동시에 사용할 수 없다. `overwrite: true`는 정확히 일치하는
run 디렉터리와 해당 run의 `.json`, `.pt`만 삭제하며 `output_dir` 자체나 다른 run은
삭제하지 않는다.

## 3. Epoch별 로그

각 epoch가 끝날 때 다음 값을 `metrics.jsonl`에 한 줄씩 즉시 기록한다.

- epoch 번호
- train loss
- validation loss
- validation token accuracy
- classifier 모델의 validation exact match
- 해당 epoch에서 사용한 learning rate
- epoch 소요 시간

Checkpoint를 먼저 원자적으로 저장한 뒤 로그를 기록한다. Resume할 때는
checkpoint에 포함된 전체 history를 기준으로 `metrics.jsonl`을 원자적으로 다시
작성한다. 따라서 checkpoint와 로그 저장 사이에 프로세스가 중단되어도 누락되거나
중복된 epoch 행이 다음 resume에서 자동으로 정리된다. 로그 기록 자체도 매 epoch마다
flush하고 디스크 동기화를 수행한다.

예시:

```json
{"epoch": 1, "train_loss": 1.61, "validation": {"loss": 1.60, "token_accuracy": 0.22}, "learning_rate": 0.0003, "epoch_seconds": 12.4}
```

## 4. YAML 설정 연결

기존에는 YAML에 작성돼 있어도 실제 trainer에서 무시되던 설정들이 있었다. 현재는
다음 항목이 실제 command-line argument와 학습 코드에 연결된다.

- `training.optimizer`
- `training.scheduler`
- `training.scheduler.warmup_epochs`
- `training.scheduler.min_lr`
- `training.seeds`
- `validation.ratio`
- `checkpointing.save_every`
- `checkpointing.resume`
- `checkpointing.overwrite`
- `performance.amp`
- `performance.num_workers`
- `performance.pin_memory`
- `performance.persistent_workers`
- `evaluation.batch_size`
- `evaluation.splits`
- `evaluation.metrics`

특히 `training.seeds: [0, 1, 2]`는 더 이상 첫 번째 seed만 사용하지 않는다. YAML
진입점으로 실행하면 세 seed를 순차적으로 모두 학습하고 각각 독립된 결과와
checkpoint를 저장한다.

```bash
python main.py --config configs/basic_model.yaml --device cuda
python main.py --config configs/looped_model.yaml --device cuda
```

생성되는 기본 run 이름:

```text
direct-sinusoidal-seed0
direct-sinusoidal-seed1
direct-sinusoidal-seed2
recurrent-sinusoidal-seed0-adaptive
recurrent-sinusoidal-seed1-adaptive
recurrent-sinusoidal-seed2-adaptive
```

## 5. AMP, Scheduler, DataLoader

### AMP

CUDA 학습에서 mixed precision과 GradScaler를 사용할 수 있다.

```yaml
performance:
  amp: true
```

AMP를 요청해도 CPU 환경에서는 자동으로 비활성화된다. Checkpoint에는 GradScaler
상태도 포함되므로 AMP 학습도 중단 지점부터 이어갈 수 있다.

### Optimizer

지원 optimizer:

- `AdamW`
- `Adam`
- `SGD` (`momentum=0.9`)

```yaml
training:
  optimizer: AdamW
```

### Scheduler

지원 scheduler:

- `none`
- `cosine`
- `linear`

```yaml
training:
  scheduler:
    name: cosine
    warmup_epochs: 0
    min_lr: 0.0
```

Scheduler는 매 epoch 종료 후 갱신되며 상태가 checkpoint에 저장된다.

### DataLoader

```yaml
performance:
  num_workers: 2
  pin_memory: true
  persistent_workers: true
```

- CUDA 환경에서는 pinned memory와 non-blocking tensor transfer 사용
- worker 프로세스를 epoch 사이에 유지 가능
- epoch별 shuffle 순서는 `seed + epoch`으로 결정되어 resume 이후에도 재현 가능

## 변경된 파일

- `src/original/experiment.py`: validation, 학습 루프, AMP, scheduler, checkpoint,
  resume, epoch 로그 구현
- `src/trainer.py`: YAML 설정 연결과 multi-seed 실행 구현
- `main.py`: single-seed와 multi-seed 결과 출력 지원
- `configs/basic_model.yaml`: 새 학습 인프라 기본 설정 추가
- `configs/looped_model.yaml`: 새 학습 인프라 기본 설정 추가
- `configs/original.yaml`: 공통 학습 인프라 설정 추가
- `tests/test_original.py`: validation, checkpoint, 설정 연결 테스트 추가
- `docs/TRAINING.md`: 실행 및 resume 사용법 추가

## 추가 안전성 보완

장시간 학습 전 점검에서 발견된 세 항목을 추가로 반영했다.

1. **로그 정합성:** checkpoint history를 기준으로 resume 시 `metrics.jsonl` 복구
2. **충돌 방지:** 동일 run은 `resume` 또는 명시적인 `overwrite` 없이는 시작 차단
3. **공정한 validation:** `n_swaps`별 층화 분할로 seed 간 길이 분포 차이 축소

고정 `run_name`과 여러 seed를 함께 사용하면 각 run 이름에 `-seedN`을 자동으로
붙여 seed끼리 결과를 덮어쓰지 않도록 했다.

## 검증 결과

로컬에서 다음 검증을 완료했다.

- 전체 자동 테스트: `20 passed`
- Direct Transformer smoke training: 통과
- Recurrent Transformer 및 adaptive KL smoke training: 통과
- epoch 1 checkpoint에서 epoch 2로 resume: 통과
- optimizer와 scheduler 상태 복원: 확인
- YAML의 seed 0, 1, 2 순차 실행: 통과
- 각 seed의 독립된 JSON 결과 생성: 확인
- `git diff --check`: 통과

Smoke test는 코드 경로와 저장 기능을 확인하기 위한 소량 데이터 실행이므로, 기록된
accuracy는 최종 연구 결과로 사용하지 않는다.

## 현재 Git 상태

- commit 생성 안 함
- push 수행 안 함
- 원격 저장소 변경 없음
- 변경 사항은 로컬 working tree에만 존재
- 기준 commit: `cfb5bc5`
