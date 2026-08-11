# Original team-plan experiment track

이 경로는 확장 연구안(`src/systematic/`)을 대체하지 않는다. 팀이 처음 합의한
스코프를 동일 저장소에서 독립적으로 실행해 결과를 비교할 수 있게 보존한다.

## 고정된 연구 범위

- 인물 5명과 색 공 5개, 초기 상태는 무작위 순열
- 입력 난이도는 **전체 교환 횟수**로 정의
- 출력은 5명 모두의 최종 공 색
- ID는 학습과 같은 swap-length 범위, OOD는 x4/x8 길이
- 비교 모델은 Direct, Explicit CoT, Recurrent
- 위치 인코딩은 Sinusoidal과 RoPE를 같은 구현에서 선택
- Recurrent의 반복식은 정확히 `h = e + block(h)`
- adaptive stopping은 연속 출력 분포의 symmetric KL만 사용

## 세 모델의 공정한 입출력

Direct와 Recurrent는 기존 23-token vocabulary, 동일 SLOT 5개, 동일 shared MLP
classifier를 사용한다. Explicit CoT만 외부 작업기억을 표현하기 위해 `[BOS]`,
`[STATE]`, `[END_STATE]` 세 control token을 추가한다.

Explicit CoT는 각 교환 다음에 아래처럼 전체 상태를 쓴다.

```text
[STATE] [SLOT_윤성] 빨간색 ... [SLOT_용준] 파란색 [END_STATE]
```

학습 때는 gold trace를 teacher forcing으로 사용하지만 loss는 각 SLOT 다음의 색
예측에만 적용한다. 평가는 gold 중간 상태를 제공하지 않고 색 토큰을
autoregressive하게 생성한다. 따라서 CoT 조건의 중간 정답 누수를 막는다. 긴 OOD
trace 평가는 layer별 attention K/V cache를 사용해 이미 처리한 prefix를 다시
계산하지 않는다.

## 실행

```bash
# 빠른 구조 검증
python scripts/run_original_experiments.py --architecture direct --smoke --device cpu
python scripts/run_original_experiments.py --architecture cot --smoke --device cpu
python scripts/run_original_experiments.py --architecture recurrent --smoke --device cpu \
  --adaptive-kl-eval

# 본 실험 예시
python scripts/run_original_experiments.py --architecture direct \
  --position-encoding sinusoidal --seed 0
python scripts/run_original_experiments.py --architecture cot \
  --position-encoding sinusoidal --seed 0
python scripts/run_original_experiments.py --architecture recurrent \
  --position-encoding sinusoidal --adaptive-kl-eval --seed 0

# PE ablation은 동일 명령에서 rope로만 변경
python scripts/run_original_experiments.py --architecture recurrent \
  --position-encoding rope --adaptive-kl-eval --seed 0
```

각 run은 `runs/original/`에 checkpoint와 JSON을 저장한다. JSON은 ID/OOD 각각에
대해 slot accuracy, 5명 전체 exact match, swap 횟수별 exact match를 기록한다.
Recurrent adaptive 평가에는 평균 loop 수, halt rate, 마지막 symmetric KL도
기록한다.

3개 seed의 Basic/Looped 비교와 집계는 한 명령으로 실행한다.

```bash
python scripts/run_original_multiseed.py --seeds 0 1 2 \
  --architectures direct recurrent --adaptive-kl-eval
python scripts/plot_original_results.py runs/original/aggregate/summary.csv \
  --output-dir runs/original/figures
```

Deep supervision과 noop robustness는 주 조건과 분리한 선택형 ablation이다.

```bash
python main.py --architecture recurrent --deep-supervision-weight 0.5 \
  --noop-eval-ratio 0.5 --adaptive-kl-eval
```

## 해석 시 주의

이 경로에서 `swap length`는 context length와 state-transition depth를 분리하지
않는다. 따라서 길이 OOD 성능 차이는 원래 팀 질문에는 답하지만, systematic state
updating 자체의 증거로 단정할 수는 없다. 그 통제 실험은 `src/systematic/`의
target-depth/distractor 분리 경로에서 수행한다.
