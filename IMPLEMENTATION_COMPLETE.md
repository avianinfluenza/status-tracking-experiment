# 모델 및 실험 구현 최종 설명

## 완료 범위

초기 팀 체크리스트의 모델·실험 구현과 이후 확장 항목을 모두 실행 가능한 형태로
완성했다. 확장 기능은 기본 실험을 오염시키지 않도록 명시적 옵션으로 분리했다.

| 체크리스트 | 최종 구현 |
|---|---|
| Basic Transformer | 독립 block을 쌓는 `DirectTransformer` |
| Looped Transformer | shared block과 `h = e + block(h)`를 쓰는 `RecurrentTransformer` |
| 슬롯 분류기 연결 | 두 모델이 동일한 5-slot shared MLP classifier 사용 |
| train/evaluate/main trainer | `main.py`와 `src/original/experiment.py` |
| Basic/Looped config | `configs/basic_model.yaml`, `configs/looped_model.yaml` |
| 위치 표현·mask 테스트 | Sinusoidal/RoPE, padding mask, causal cache 테스트 |
| 3개 이상 seed 비교 | seed 0/1/2 일괄 실행기 및 6-run smoke 검증 |
| swap별 집계·그래프 | raw/summary CSV와 PNG 또는 dependency-free SVG 출력 |
| loop deep supervision | `--deep-supervision-weight` 선택형 ablation |
| CoT/teacher forcing | 누수 없는 autoregressive Explicit CoT 평가 |
| noop 강건성 | `--noop-eval-ratio`로 gold-preserving self-swap 평가 |

## 새 저장소 구조와의 통합

최신 `main`의 패키지 구성을 기준으로 데이터 코드는 `src/data/`, 모델 코드는
`src/model/`에 배치했다. 기존 모델 구현은 `src/model/state_tracking.py`에 보존하고,
팀 공통 import 경로인 `src/model/basic_transformer.py`와
`src/model/looped_transformer.py`에서 각각 Basic/Looped 구현을 노출한다.

루트 `main.py`는 두 실행 방식을 함께 지원한다. `--config`가 있으면
`src/trainer.py`가 팀 YAML 설정을 기존 실험 실행기의 인자로 변환하고, 없으면 기존
직접 CLI를 그대로 사용한다. 따라서 새 구조를 따르면서도 기존 실험 명령과 결과
형식을 유지한다.

## 모델 구조

### Basic/Direct

Basic 모델은 서로 다른 파라미터를 가진 Transformer block `L`개를 통과한다. 입력
본문 뒤의 `[SLOT_인물]` 5개 위치를 모아 한 개의 shared MLP classifier로 각 인물이
가진 최종 공 색을 예측한다.

### Looped/Recurrent

Looped 모델은 한 개의 Transformer block을 `T`회 공유한다. 초기 팀 설계의 반복식을
그대로 사용한다.

```text
h = e + block(h)
```

`forward_all_loops()`는 각 loop의 5-slot logits를 반환한다. 기본 학습은 마지막
loop만 감독한다. `--deep-supervision-weight W`를 0보다 크게 줄 때만 비최종 loop의
평균 CE를 `W`만큼 더한다. 따라서 deep supervision은 주 조건이 아니라 ablation이다.

Adaptive halting은 현재와 직전 5-slot 출력 분포의 symmetric KL만 사용한다.
confidence, hidden-state norm, 정답, 실제 swap length는 halting에 쓰지 않는다.

### Explicit CoT

각 swap 뒤에 5명 전체 상태를 외부 토큰으로 기록한다. 학습에서는 teacher forcing을
사용하지만 SLOT 다음의 색만 감독한다. 평가는 gold 중간 상태 없이 색을 순차 생성해
최종 상태를 얻는다. 긴 OOD 생성은 attention K/V cache로 처리한다.

## 위치 표현과 attention mask

`--position-encoding sinusoidal|rope`로 세 모델의 위치 표현을 바꾼다. Direct와
Recurrent는 padding을 attention key에서 제외하며, 논리적 token position을 사용해
같은 샘플이 배치의 다른 길이에 영향을 받지 않게 한다. CoT는 causal mask를 쓰고,
cached decoding이 전체 causal forward와 같은 출력을 내는지 테스트한다.

## Noop 강건성 평가

`--noop-eval-ratio R`은 ID 예제의 원래 swap 순서를 보존하면서 `a ↔ a` self-swap을
원래 swap 수 대비 `R` 비율로 삽입한다. self-swap은 symbolic state를 바꾸지 않으므로
기존 gold label을 그대로 유지한다. 결과는 `id_test_noop_R` split으로 별도 기록한다.
기본값 0에서는 이 split을 만들지 않는다.

## 실행 진입점

단일 모델:

```bash
python main.py --architecture direct --seed 0
python main.py --architecture recurrent --seed 0 --adaptive-kl-eval
python main.py --architecture cot --seed 0
```

팀 공통 YAML config 진입점:

```bash
python main.py --config configs/basic_model.yaml --smoke --device cpu
python main.py --config configs/looped_model.yaml --smoke --device cpu
```

선택형 확장:

```bash
python main.py --architecture recurrent --seed 0 \
  --deep-supervision-weight 0.5 \
  --noop-eval-ratio 0.5 \
  --adaptive-kl-eval
```

Basic/Looped 3-seed 본 비교:

```bash
python scripts/run_original_multiseed.py \
  --seeds 0 1 2 \
  --architectures direct recurrent \
  --position-encoding sinusoidal \
  --adaptive-kl-eval
```

위 명령은 모델별 JSON/checkpoint와 다음 파일을 자동 생성한다.

```text
runs/original/
├── multiseed_manifest.json
└── aggregate/
    ├── raw_long.csv
    └── summary.csv
```

그래프:

```bash
python scripts/plot_original_results.py \
  runs/original/aggregate/summary.csv \
  --output-dir runs/original/figures
```

`matplotlib`이 있으면 PNG를, 없으면 외부 의존성 없이 SVG를 만든다. 그래프의 x축은
전체 swap 수, y축은 5명 전체 exact match이며 seed 평균과 표준편차를 표시한다.

이미 따로 실행한 결과 JSON을 합칠 때는 다음 명령을 사용한다.

```bash
python scripts/aggregate_original_results.py runs/original/*.json \
  --output-dir runs/original/aggregate
```

## 결과 형식

각 단일 run JSON은 다음 정보를 가진다.

- architecture, position encoding, parameter 수, seed
- train loss와 학습 시간
- ID/OOD x4/OOD x8의 slot accuracy와 5-person exact match
- 각 split의 swap 횟수별 exact match
- KL halting 사용 시 평균 loop, halt rate, 마지막 symmetric KL
- deep-supervision weight와 noop ratio

`raw_long.csv`는 model × seed × split × swap length × metric 단위를 보존한다.
`summary.csv`는 같은 조건의 seed 평균, 표본 표준편차, 최솟값, 최댓값과 seed 수를
기록한다.

## 검증 결과

- 전체 기존·신규 테스트 `36 passed`
- Basic, Looped, Explicit CoT 단일 smoke 통과
- Sinusoidal/RoPE와 padding/causal attention 검증 통과
- deep-supervision 학습 경로와 noop gold 보존 검증 통과
- Direct/Recurrent × seed 0/1/2의 6-run smoke 비교 통과
- raw CSV, summary CSV, swap-length SVG 생성 통과

Smoke run은 작은 모델, 16개 학습 샘플, 1 epoch의 기능 검증이다. 여기서 얻은
accuracy는 연구 결과가 아니다. 논문·발표 결과에는 `--smoke`를 제거한 본 실험과
여러 seed의 집계만 사용해야 한다.

## 해석 경계

초기 팀 경로는 전체 swap length가 길어질 때의 일반화를 측정한다. 이 조건에서는
context length와 target transition depth가 함께 변할 수 있다. 따라서 이 결과만으로
systematic state updating을 입증하면 안 된다. 두 변수를 분리한 주장은
`src/systematic/`의 확장 연구 경로로 검증한다.
