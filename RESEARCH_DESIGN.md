# 연구 질문–코드 정합성 점검

## 결론

기존 ball-swap 구현은 recurrent weight sharing 자체는 구현했지만, 연구 질문을
검증하는 실험으로는 불충분했다. 전체 교환 횟수를 난이도로 사용하고 모든 인물의
상태를 동시에 예측했기 때문에 다음 두 변수가 섞여 있었다.

- 정답에 실제로 필요한 target entity의 state-transition depth
- 정답과 무관한 distractor 및 전체 context length

또한 기존 recurrent 기본값에는 step conditioning과 residual damping이 포함되어
있어, 동일 update operator를 그대로 반복하는 R0의 효과를 분리할 수 없었다.
학습 loop보다 큰 inference loop도 허용하지 않아 `D ↑ => K*(D) ↑`를 시험할 수
없었다.

현재 실행 우선순위는 `src/original/`의 ball-swap을 R0 recurrent 구현으로 다시
검증하는 것이다. `src/systematic/`은 위 confound를 분리하기 위한 controlled
object-location 확장 경로로 보존하고, ball-swap 결과를 확인한 뒤 보조 실험으로
실행한다.

## 가설과 직접 대응하는 실험

| 가설/질문 | 조작 | 통제 | 코드 |
|---|---|---|---|
| H1: ID fitting은 비슷한가 | Standard vs Recurrent | D=1–8 | `evaluate` (E0) |
| H2: depth OOD에서 recurrence가 유리한가 | D=10,12,16,20,24,32 | distractor=8 | E1 depth loaders |
| H3: 더 깊은 문제에 더 많은 loop가 필요한가 | D × K | 동일 checkpoint | `loop_depth_sweep` (E2) |
| H4: context length 효과인가 | D=2,4,8,12,16,20 | total events=24 | `matched_length_grid` (E3) |
| retrieval/distractor 효과인가 | distractor=0,4,8,16,32,64 | D=8 | E4 distractor sweep |
| 언어/어휘에 일반화하는가 | held-out template/lexicon | symbolic transition | E5 OOD splits |
| H5: 상태가 반복 중 체계적으로 형성되는가 | loop별 CLS state | 별도 probe split | E6 probe/trajectory matrix |
| 어떤 recurrent 설계가 필요한가 | fixed/random/LoopEmb/sharing ratio | 동일 protocol | E7 ablation |

H2만으로 “systematic state updating”을 결론 내리지 않는다. 최소한 H3와 E4가
같이 성립해야 추가 recurrence가 단순한 parameter efficiency나 긴 문맥 retrieval이
아니라 state-transition computation budget으로 사용된다는 강한 해석이 가능하다.

## 모델 조건

- **Standard**: 서로 독립인 `L`개 block
- **R0 Recurrent**: block 하나를 `T`번 정확히 반복. loop conditioning 없음,
  residual scale 1.0
- **R1**: R0 + learned loop embedding
- **Residual ablation**: R0 + residual scale `< 1`
- **Random-loop ablation**: batch마다 train loop 수를 표본 추출
- **Untied iterative control**: 동일 iteration API에서 각 step의 block은 독립
- **Partial sharing**: `--recurrent-blocks 2` 또는 `4`로 A/B, A/B/C/D 순환

Adaptive halting은 고정-loop 비교의 주 조건에 넣지 않는다. overthinking과
계산량 절감을 확인하는 후속 ablation으로만 사용해야 한다. 먼저 loop sweep
전체를 저장해 accuracy와 마지막 update ratio를 분리해서 해석한다.

## 공정 비교

한 표에서 서로 다른 질문을 섞지 않는다.

1. Compute-matched: Standard `L=T` vs Recurrent `T`
2. Parameter-matched: width를 조절해 총 trainable parameter를 맞춘 별도 표
3. Sharing continuum(후속): `1/T`, `2/T`, `4/T`, `T/T` unique blocks

각 조건은 seed 0, 1, 2 이상으로 실행하고 평균±표준편차를 보고한다. 단일 seed의
smoke run은 구현 검증일 뿐 가설 검정 결과가 아니다.

집계기는 bootstrap confidence interval, paired seed accuracy gap, paired effect
size, sign-flip test와 depth별 Holm 보정, `Spearman ρ(D,K*)`를 계산한다. E0의 모든
depth가 사전 기준 95%를 넘지 못하면 결과 파일에 `ood_conclusions_allowed=false`를
기록한다.

## 실행

```bash
# Current ball-swap R0 main condition
python scripts/run_original_experiments.py \
  --architecture recurrent-r0 --num-loops 6 --seed 0 \
  --eval-loop-counts 1 2 4 6 8 12 16 24

# Current ball-swap 3-seed comparison
python scripts/run_original_multiseed.py \
  --seeds 0 1 2 \
  --architectures direct recurrent-r0 \
  --eval-loop-counts 1 2 4 6 8 12 16 24

# Later controlled object-location R0 condition
# R0 main condition
python scripts/run_systematic_experiments.py \
  --architecture recurrent --train-loops 6 \
  --loop-counts 1 2 4 6 8 12 16 24 32 --seed 0

# Standard compute-matched condition
python scripts/run_systematic_experiments.py \
  --architecture standard --num-layers 6 --seed 0

# 위 결과 JSON의 parameters를 recurrent budget으로 사용
python scripts/run_systematic_experiments.py \
  --architecture recurrent --target-parameters <STANDARD_PARAMS> --seed 0

# R1 and training-procedure ablations
python scripts/run_systematic_experiments.py \
  --architecture recurrent --loop-conditioning learned --seed 0
python scripts/run_systematic_experiments.py \
  --architecture recurrent --random-loops --seed 0

# End-to-end validation only
python scripts/run_systematic_experiments.py --smoke
```

## 해석 전 사전 기준

- H1: 두 모델 모두 ID task를 충분히 학습해야 H2 비교를 진행한다.
- H2: train maximum보다 큰 target depth에서 recurrent degradation이 더 느린지 본다.
- H3: `K*(D)`가 depth와 함께 증가하는지 보고, 너무 큰 K에서 하락하는
  overthinking도 함께 기록한다.
- H4: 같은 total event 수에서 target depth만 늘려도 차이가 유지되어야 한다.
- H5: loop index와 event index의 1:1 대응을 가정하지 않고, 전체
  loop×symbolic-trajectory 행렬에서 emergence만 검사한다. Linear probe는 encoder를
  freeze한 뒤 별도 train/probe split으로 학습한다. 같은 classifier로 intermediate
  state를 읽는 행렬에는 추가 intermediate supervision을 사용하지 않는다.
