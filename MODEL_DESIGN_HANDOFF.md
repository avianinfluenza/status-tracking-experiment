# 모델 설계 작업 인수인계

## 작업 목적과 범위

5인 공 교환 상태추적 과제의 초기 팀 연구안에 맞춰 Basic/Direct, Explicit CoT,
Looped/Recurrent 모델과 학습·평가 경로를 구현했다.

- 기존 JSONL split과 고정 23-token vocabulary의 순서·ID는 변경하지 않았다.
- 출력은 모든 모델에서 5명 전체의 최종 공 색을 예측한다.
- 모델 비교에 필요한 trainer, config, 테스트, 결과 집계 도구까지 연결했다.
- smoke run은 실행 가능성만 확인하며 연구 결과로 사용하지 않는다.

## 구현된 모델

### Direct/Standard

서로 다른 가중치를 가진 Transformer block을 `L`개 쌓는다. 본문 뒤의 5개 SLOT
hidden state를 모아 shared classifier로 각 인물의 최종 색을 동시에 예측한다.

### Recurrent/Looped

하나의 shared Transformer block을 `T`회 반복하며 다음 update를 사용한다.

```text
h = e + block(h)
```

Direct와 동일한 SLOT 및 classifier를 사용한다. Adaptive stopping은 현재 출력과
직전 출력의 symmetric KL만 사용하며, confidence·hidden norm·gold swap length는
halting 결정에 사용하지 않는다.

### Explicit CoT

각 교환 다음에 5명의 전체 상태를 외부 토큰으로 기록한다. 학습에서는 gold trace를
teacher forcing으로 제공하되 각 SLOT 다음 색 토큰에만 loss를 적용한다. 평가에서는
gold 중간 상태 없이 autoregressive하게 생성하며, 긴 trace는 attention K/V cache로
처리한다.

### 위치 표현

세 모델 모두 `sinusoidal`과 `rope`를 같은 옵션으로 선택할 수 있다.

## 파일별 역할

| 파일 | 역할 |
|---|---|
| `src/data/` | vocabulary, collate, dataset, 독립 데이터 검증기 |
| `src/model/classifier.py` | 공통 slot classifier |
| `src/model/basic_transformer.py` | Basic/Direct 모델 import 경로 |
| `src/model/looped_transformer.py` | Looped/Recurrent 모델 import 경로 |
| `src/original/model.py` | Direct, Explicit CoT, Recurrent, Sinusoidal/RoPE, KL halting |
| `src/original/data.py` | CoT trace 변환, symbolic replay, noop ablation |
| `src/original/experiment.py` | 학습, ID/OOD 평가, 결과 및 checkpoint 저장 |
| `src/trainer.py` | YAML config를 실험 실행기로 연결 |
| `main.py` | YAML config 또는 직접 CLI를 선택하는 통합 진입점 |
| `scripts/run_original_multiseed.py` | 다중 seed Basic/Looped 비교 |
| `tests/test_original.py` | 반복식, 위치 표현, KL halting, CoT 누수·cache 검증 |

## 검증 명령

```bash
python -m pip install -e '.[dev]'
python -m src.data.verify
pytest
```

빠른 모델 실행:

```bash
python main.py --config configs/basic_model.yaml --smoke --device cpu
python main.py --config configs/looped_model.yaml --smoke --device cpu
python scripts/run_original_experiments.py --architecture cot --smoke --device cpu
```

## 팀에서 결정할 사항

1. Sinusoidal과 RoPE 중 하나를 고정할지, 둘 다 비교할지 결정한다.
2. Direct layer 수와 Recurrent loop 수의 비교 조건을 고정한다.
3. KL threshold는 ID validation에서만 선택한다.
4. 본 실험은 최소 seed 3개로 실행하고 평균과 표준편차를 보고한다.
5. 자동 생성된 swap-length별 성능 곡선을 검토한다.
