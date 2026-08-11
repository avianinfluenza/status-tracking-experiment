# 모델 설계 작업 인수인계

## 1. 작업 목적

모델 담당 범위에서 두 가지 연구 경로를 모두 실행할 수 있도록 정리했다.

1. **확장 연구안**: target transition depth와 distractor/context length를 분리해
   recurrent depth가 systematic state updating에 사용되는지 검증한다.
2. **초기 팀 연구안**: 5명의 공 교환 문제에서 전체 swap length를 기준으로
   Standard/Direct, Explicit CoT, Recurrent의 ID/OOD 길이 일반화를 비교한다.

확장 연구안이 팀의 최종 선택을 받지 못하더라도 초기 연구안을 그대로 실행할 수
있게 두 경로를 분리했다. 확장 경로는 `src/systematic/`, 초기 경로는
`src/original/`이다. 두 경로를 한 실험의 결과처럼 섞어 해석하면 안 된다.

## 2. 이번 작업의 범위

이번 변경은 **모델 설계와 모델 통합 검증**에 한정했다.

- 기존 JSONL 데이터셋과 split을 수정하지 않았다.
- 고정 23-token vocabulary의 순서와 ID를 수정하지 않았다.
- 팀원이 만든 shared MLP classifier 구현을 수정하지 않고 재사용했다.
- 기존 데이터 생성기와 advanced/systematic 구현을 수정하지 않았다.
- 본 실험 결과를 주장하지 않았다. 수행한 smoke run은 실행 가능성 검증용이다.

모델을 실제 데이터에 연결해 검증하기 위해 실행기, 설정, 테스트, 문서를 함께
추가했다. 이는 별도 데이터 연구나 결과 분석을 대신하기 위한 코드가 아니다.

## 3. 초기 팀 연구안 구현

### 공통 문제 정의

- 인물 5명, 색 공 5개
- 초기 상태는 무작위 순열
- 입력 난이도는 전체 교환 횟수
- 출력은 5명 모두의 최종 공 색
- ID는 train과 같은 swap-length 범위
- OOD는 기존 `ood_x4`, `ood_x8` split
- 주요 지표는 slot accuracy와 5명 전체 exact match

### Direct/Standard

서로 다른 가중치를 가진 Transformer block을 `L`개 쌓는다. 본문 뒤의 5개 SLOT
hidden state를 모아 기존 shared classifier로 각 인물의 최종 색을 동시에 예측한다.

### Recurrent/Looped

하나의 shared Transformer block을 `T`회 반복한다. 반복식은 초기 팀 설계에 맞춰
아래 형태로 고정했다.

```text
h = e + block(h)
```

여기서 `e`는 원래 token embedding이다. Direct와 같은 SLOT 및 classifier를
사용하므로 출력 head 차이가 비교를 교란하지 않게 했다.

Adaptive stopping은 현재 출력과 직전 출력의 symmetric KL만 사용한다. confidence,
hidden update norm, gold swap length 등 다른 신호는 halting 결정에 들어가지 않는다.
`min_loops`와 `patience`는 KL 조건을 적용하는 시점과 연속 충족 횟수만 제어한다.

### Explicit CoT

각 교환 다음에 5명의 전체 상태를 외부 토큰으로 기록한다.

```text
[STATE] [SLOT_윤성] 빨간색 ... [SLOT_용준] 파란색 [END_STATE]
```

학습에서는 gold trace를 teacher forcing으로 제공하되 각 SLOT 다음 색 토큰에만
loss를 적용한다. 평가에서는 gold 중간 상태를 주지 않고 모델이 상태를
autoregressive하게 생성한다. 따라서 Explicit CoT만 중간 정답을 입력으로 받는
누수를 막았다. 긴 OOD trace는 attention K/V cache를 이용해 생성한다.

### 위치 인코딩

세 모델 모두 `sinusoidal`과 `rope`를 동일한 CLI 옵션으로 선택할 수 있다. 위치
인코딩 이외의 설정을 그대로 둔 채 비교할 수 있다.

## 4. 파일별 역할

| 파일 | 역할 |
|---|---|
| `src/data/` | vocabulary, collate, dataset, 독립 데이터 검증기 |
| `src/model/state_tracking.py` | 확장 연구안의 Standard/Recurrent 및 안정화 ablation |
| `src/model/classifier.py` | 두 연구 경로가 공유하는 slot classifier |
| `src/model/basic_transformer.py` | 팀 공통 Basic 모델 import 경로 |
| `src/model/looped_transformer.py` | 팀 공통 Looped 모델 import 경로 |
| `src/trainer.py` | 팀 YAML 설정을 초기 연구안 실행기로 연결하는 adapter |
| `src/original/model.py` | Direct, Explicit CoT, Recurrent, Sinusoidal/RoPE, KL halting |
| `src/original/data.py` | 기존 row의 CoT trace 변환과 symbolic replay |
| `src/original/experiment.py` | 학습, ID/OOD 평가, 결과 및 checkpoint 저장 |
| `scripts/run_original_experiments.py` | 초기 연구안 실행 CLI |
| `configs/original.yaml` | 초기 연구안 기준 설정 |
| `configs/basic_model.yaml` | Basic 모델의 명시적 기준 설정 |
| `configs/looped_model.yaml` | Looped 모델과 KL halting 기준 설정 |
| `main.py` | YAML 설정 또는 기존 CLI를 선택하는 통합 진입점 |
| `tests/test_original.py` | 반복식, PE, KL halting, CoT 누수·cache 검증 |
| `ORIGINAL_PLAN.md` | 초기 연구안의 실행법과 해석상 한계 |
| `IMPLEMENTATION_COMPLETE.md` | 최종 기능, 다중 seed, 집계·그래프 실행법 |

## 5. 실행 방법

먼저 전체 회귀 테스트와 데이터 검증을 수행한다.

```bash
python -m pip install -e '.[dev]'
python -m src.data.verify
pytest
```

각 모델의 빠른 실행 확인:

```bash
python scripts/run_original_experiments.py --architecture direct --smoke --device cpu
python scripts/run_original_experiments.py --architecture cot --smoke --device cpu
python scripts/run_original_experiments.py --architecture recurrent --smoke --device cpu \
  --adaptive-kl-eval
```

본 실험에서는 `--smoke`를 제거하고 seed와 위치 인코딩을 명시한다. 출력은 기본적으로
`runs/original/`에 JSON과 checkpoint로 저장된다.

## 6. 현재 검증 상태

- 전체 테스트 `36 passed`
- 기존 데이터 pipeline 검증 전부 통과
- Direct, Explicit CoT, Recurrent의 1-epoch smoke 학습 통과
- 세 모델의 ID/OOD x4/OOD x8 평가와 결과 저장 통과
- Recurrent의 RoPE 및 KL-only adaptive 평가 통과
- Explicit CoT의 cached decoding과 전체 causal forward 출력 일치 검증 통과

Smoke accuracy는 극소량 데이터와 1 epoch로 얻은 값이므로 연구 결과로 해석하지
않는다.

## 7. 구현 완료 후 팀에서 결정할 사항

1. 초기 연구안을 본 실험으로 유지할지, 확장 연구안을 주 실험으로 채택할지 결정한다.
2. Sinusoidal과 RoPE 중 하나를 고정할지, PE ablation으로 둘 다 실행할지 결정한다.
3. Direct `L`과 Recurrent `T`를 같게 둔 compute-depth 비교 외에 parameter-matched
   비교를 추가할지 결정한다.
4. 본 결과를 만들 때 smoke가 아닌 설정으로 최소 seed 3개를 실행한다.
5. KL threshold는 ID validation으로만 선택하고 OOD test 결과를 보고 조정하지 않는다.
6. 자동 생성된 평균·표준편차와 swap-length별 성능 곡선을 검토한다.

초기 연구안의 결과는 **swap/context length 일반화**에 대한 근거다. 이 결과만으로
state-transition depth의 systematic updating을 입증했다고 해석하면 안 된다.
target depth와 distractor를 분리한 주장은 확장 경로의 통제 실험으로 검증해야 한다.
