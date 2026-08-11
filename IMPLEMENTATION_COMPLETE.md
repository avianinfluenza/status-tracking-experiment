# 모델 및 실험 구현 설명

## 완료 범위

| 항목 | 구현 내용 |
|---|---|
| Basic Transformer | 독립 block을 쌓는 `DirectTransformer` |
| Looped Transformer | shared block과 `h = e + block(h)`를 쓰는 `RecurrentTransformer` |
| 슬롯 분류기 | 두 모델이 동일한 5-slot shared MLP classifier 사용 |
| Explicit CoT | 5명 전체 상태 trace와 누수 없는 autoregressive 평가 |
| train/evaluate/main | `main.py`, `src/trainer.py`, `src/original/experiment.py` |
| Basic/Looped config | `configs/basic_model.yaml`, `configs/looped_model.yaml` |
| 위치 표현·mask | Sinusoidal/RoPE, padding mask, causal cache 테스트 |
| 다중 seed 비교 | seed 0/1/2 일괄 실행기 |
| 결과 분석 | raw/summary CSV와 swap-length별 PNG 또는 SVG |
| 선택형 ablation | loop deep supervision과 noop robustness |

## 모델 구조

### Basic/Direct

서로 다른 파라미터를 가진 Transformer block `L`개를 통과한다. 입력 본문 뒤의
`[SLOT_인물]` 5개 위치를 모아 shared MLP classifier로 최종 공 색을 예측한다.

### Looped/Recurrent

한 개의 Transformer block을 `T`회 공유하며 다음 update를 사용한다.

```text
h = e + block(h)
```

기본 학습은 마지막 loop만 감독한다. `--deep-supervision-weight`를 0보다 크게 줄
때만 비최종 loop의 평균 CE를 추가한다. Adaptive halting은 현재와 직전 5-slot 출력
분포의 symmetric KL만 사용한다.

### Explicit CoT

각 swap 뒤에 5명 전체 상태를 외부 토큰으로 기록한다. 학습에서는 teacher forcing을
사용하지만 SLOT 다음의 색만 감독한다. 평가는 gold 중간 상태 없이 색을 순차 생성해
최종 상태를 얻으며 attention K/V cache를 사용한다.

## 위치 표현과 attention mask

`--position-encoding sinusoidal|rope`로 위치 표현을 선택한다. Direct와 Recurrent는
padding key를 attention에서 제외한다. CoT는 causal mask를 사용하고 cached decoding과
전체 causal forward가 같은 출력을 내는지 테스트한다.

## 실행 진입점

```bash
# 직접 CLI
python main.py --architecture direct --seed 0
python main.py --architecture recurrent --seed 0 --adaptive-kl-eval
python main.py --architecture cot --seed 0

# YAML config
python main.py --config configs/basic_model.yaml --smoke --device cpu
python main.py --config configs/looped_model.yaml --smoke --device cpu
```

3-seed 비교:

```bash
python scripts/run_original_multiseed.py \
  --seeds 0 1 2 \
  --architectures direct recurrent \
  --position-encoding sinusoidal \
  --adaptive-kl-eval
```

결과 그래프:

```bash
python scripts/plot_original_results.py \
  runs/original/aggregate/summary.csv \
  --output-dir runs/original/figures
```

각 run은 architecture, position encoding, parameter 수, seed, train loss와 함께
ID/OOD의 slot accuracy, 5-person exact match, swap 횟수별 exact match를 기록한다.
KL halting 사용 시 평균 loop, halt rate, 마지막 symmetric KL도 저장한다.

## 검증 기준

- 전체 단위 테스트 통과
- 데이터 독립 검증 통과
- Basic, Looped, Explicit CoT smoke 학습 통과
- Direct/Recurrent × seed 0/1/2 비교 실행 통과
- raw CSV, summary CSV, swap-length 그래프 생성 통과

Smoke run의 accuracy는 연구 결과가 아니다. 최종 결과에는 `--smoke`를 제거한 본
실험과 여러 seed의 집계만 사용한다.
