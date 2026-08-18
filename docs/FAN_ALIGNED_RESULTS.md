# Fan-Aligned Length Generalization Results

이 문서는 Fan-style recurrent 조건과 fixed-depth basic Transformer baseline의
현재 결과를 누적한다. 초기 2x2 및 width scaling 표는 seed 0 기준이고, 핵심
조건(`Atomic + NoPE` recurrent, Basic L6/L10)은 seed 0/1/2 반복 결과까지
포함한다. 목적은 기존 full-sequence recurrent 실험의 실패가 recurrent
computation 자체의 실패인지, 아니면 입력 표현과 위치 정보에 의존한 실패인지
분리하는 것이다.

## 공통 설정

기본 Fan-aligned recurrent 조건은 다음 설정을 공유한다. Width/depth scaling
조건은 아래 개별 표에서 model size와 layers를 별도로 표시한다.

| 항목 | 값 |
|---|---|
| Architecture | `fan-recurrent` |
| 학습 길이 | 2~10 swaps |
| 학습 방식 | online training, 100000 steps |
| Curriculum | 2~10 swaps, 1000 steps/length |
| Loop mapping | `swaps_per_loop=1`, 평가 시 기본 K=N |
| Supervision | final CE only (`deep_supervision_weight=0`) |
| Model size | `d_model=128`, heads 4, `d_ff=512`, layers 1 |
| Dropout | 0.0 |
| Seed | 0 for initial 2x2/scaling; 0/1/2 for core seed sweep |

주요 평가는 다음 네 구간을 본다.

| Split | Swap 범위 | 의미 |
|---|---:|---|
| ID | 2~10 | 학습 범위 내 일반화 |
| Boundary | 11~19 | 학습 직후 외삽 붕괴 지점 |
| OOD x4 | 20~40 | 중간 길이 외삽 |
| OOD x8 | 40~80 | 긴 길이 외삽 |

아래 표의 값은 `exact_match / slot_accuracy` 형식이다.

## Representation x Positional Encoding 2x2

| Input format | Position encoding | Easy N=2 | ID 2~10 K=N | Boundary 11~19 K=N | Boundary 11~19 K=24 | OOD 20~40 K=N | OOD 20~40 K=64 | OOD 40~80 K=N |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Template | NoPE | 0.000 / 0.205 | n/a | n/a | n/a | n/a | n/a | n/a |
| Template | Sinusoidal | 1.000 / 1.000 | 1.000 / 1.000 | 0.003 / 0.274 | 0.004 / 0.272 | 0.000 / 0.213 | 0.000 / 0.208 | 0.000 / 0.202 |
| Atomic | NoPE | 0.998 / 1.000 | 1.000 / 1.000 | 0.916 / 0.978 | 0.980 / 0.994 | 0.042 / 0.358 | 0.142 / 0.470 | 0.000 / 0.191 |
| Atomic | Sinusoidal | pending | pending | 0.190 / 0.477 | 0.251 / 0.530 | 0.000 / 0.201 | 0.000 / 0.200 | pending |

`Atomic + Sinusoidal`의 boundary와 OOD 20~40 K=64 값은 CUDA run
`fan-atomic-sinusoidal-curr2to10-seed0__20260816-095039`에서 나온 결과다.
같은 run의 `result.json` 값이 제공되면 ID 2~10과 OOD 40~80 K=N 칸을 채운다.

### Boundary Breakdown

Boundary 11~19에서 가장 중요한 비교는 `Atomic + NoPE`와
`Atomic + Sinusoidal`이다. 두 조건은 입력 표현이 같고 positional encoding만
다르다.

| Swaps | Atomic + NoPE, K=N | Atomic + Sinusoidal, K=N | Atomic + NoPE, K=24 | Atomic + Sinusoidal, K=24 |
|---:|---:|---:|---:|---:|
| 11 | 1.00 | 1.00 | n/a | 0.98 |
| 12 | 1.00 | 0.59 | n/a | 0.89 |
| 13 | 1.00 | 0.07 | n/a | 0.31 |
| 14 | 0.99 | 0.03 | n/a | 0.07 |
| 15 | 0.98 | 0.01 | n/a | 0.00 |
| 16 | 0.93 | 0.00 | n/a | 0.00 |
| 17 | 0.90 | 0.00 | n/a | 0.00 |
| 18 | 0.85 | 0.00 | n/a | 0.01 |
| 19 | 0.59 | 0.01 | n/a | 0.00 |

`Atomic + NoPE`는 boundary 전반에서 높은 exact match를 유지한다. 반면
`Atomic + Sinusoidal`은 11~12 swaps까지만 버티고 13 swaps부터 급격히
무너진다. K=24로 추가 계산을 주면 12~13 swaps는 개선되지만 14 swaps 이후의
붕괴는 해결하지 못한다.

## 해석

현재 seed 0 결과는 세 가지를 분리해서 보여준다.

1. `Template + NoPE`는 가장 쉬운 N=2 diagnostic도 풀지 못했다. 자연어 template
   토큰열에서 순서와 binding을 구분할 정보가 부족하다.
2. `Template + Sinusoidal`은 ID는 완전히 풀지만 boundary부터 거의 chance로
   떨어진다. 위치 정보를 이용해 학습 범위 안에서는 최적화되지만 길이 외삽에는
   실패한다.
3. `Atomic + NoPE`는 Fan-style 조건에 가장 가깝고, boundary 11~19에서 강한
   일반화를 보인다. OOD 20~40에서도 exact는 낮지만 slot accuracy와 K=64
   개선이 남아 있어 완전한 random guessing은 아니다.
4. `Atomic + Sinusoidal`은 representation을 구조화해도 positional encoding이
   들어가면 OOD 20~40에서 다시 chance로 붕괴한다.

따라서 현재 결론은 다음과 같다.

> Recurrent computation 자체가 무력한 것은 아니다. Fan-aligned recurrence가
> 작동하려면 입력 representation이 length-invariant해야 하며, swap event를
> absolute token position에 묶는 positional encoding은 긴 transition sequence
> 외삽을 크게 방해한다.

## Main Seed Sweep: Fan-Recurrent vs Basic

`scripts/run_main_seed_and_scaling.sh` 실행 후 seed 1/2 결과를 추가했다. 아래
표는 seed 0 기존 결과와 이번 run의 seed 1/2 결과를 합친 것이다. `Best fixed K`는
해당 split의 fixed-loop sweep 중 exact match가 가장 높은 K를 고른 값이다.

| Model | Seed | ID K=N | Boundary K=N | Boundary best fixed K | OOD 20~40 K=N | OOD 20~40 best fixed K | OOD 40~80 K=N |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fan d128 L1 | 0 | 1.000 / 1.000 | 0.916 / 0.978 | 0.980 / 0.994 at K=24 | 0.042 / 0.358 | 0.142 / 0.470 at K=64 | 0.000 / 0.191 |
| Fan d128 L1 | 1 | 1.000 / 1.000 | 0.876 / 0.965 | 0.898 / 0.971 at K=20 | 0.038 / 0.335 | 0.046 / 0.324 at K=24 | 0.002 / 0.196 |
| Fan d128 L1 | 2 | 1.000 / 1.000 | 0.928 / 0.982 | 0.936 / 0.984 at K=24 | 0.058 / 0.389 | 0.088 / 0.407 at K=40 | 0.000 / 0.185 |
| Fan d128 L1 | mean | 1.000 / 1.000 | 0.906 / 0.975 | 0.938 / 0.983 | 0.046 / 0.361 | 0.092 / 0.400 | 0.001 / 0.191 |
| Basic L6 | 0 | 1.000 / 1.000 | 0.259 / 0.498 | n/a | 0.000 / 0.203 | n/a | 0.000 / 0.206 |
| Basic L6 | 1 | 1.000 / 1.000 | 0.240 / 0.482 | n/a | 0.000 / 0.203 | n/a | 0.000 / 0.198 |
| Basic L6 | 2 | 1.000 / 1.000 | 0.246 / 0.517 | n/a | 0.000 / 0.198 | n/a | 0.000 / 0.202 |
| Basic L6 | mean | 1.000 / 1.000 | 0.248 / 0.499 | n/a | 0.000 / 0.201 | n/a | 0.000 / 0.202 |
| Basic L10 | 0 | 1.000 / 1.000 | 0.280 / 0.533 | n/a | 0.000 / 0.200 | n/a | 0.000 / 0.200 |
| Basic L10 | 1 | 1.000 / 1.000 | 0.360 / 0.620 | n/a | 0.000 / 0.208 | n/a | 0.000 / 0.202 |
| Basic L10 | 2 | 1.000 / 1.000 | 0.356 / 0.598 | n/a | 0.000 / 0.201 | n/a | 0.000 / 0.190 |
| Basic L10 | mean | 1.000 / 1.000 | 0.332 / 0.584 | n/a | 0.000 / 0.203 | n/a | 0.000 / 0.197 |

Seed 반복 후에도 핵심 비교는 유지된다.

1. Fan d128 L1은 모든 seed에서 ID를 완전히 풀고, boundary 11~19에서 평균
   0.906 exact를 유지한다. fixed K를 적절히 주면 boundary 평균은 0.938까지
   오른다.
2. Basic L6/L10은 모든 seed에서 ID를 완전히 풀지만, boundary 평균은 각각
   0.248, 0.332에 머문다. OOD 20~40 exact는 모든 seed에서 0.000이다.
3. Fan d128 L1의 OOD 20~40 exact는 낮고 seed variance가 있다. 그래도 평균
   0.046 K=N, best fixed 평균 0.092로 Basic의 0.000과 분리된다.
4. OOD 40~80은 Fan도 Basic도 사실상 chance다. 따라서 현재 결과는 "boundary와
   20~40 초반 외삽 경계를 오른쪽으로 민다"까지는 말할 수 있지만, 40~80
   systematic extrapolation은 아직 아니다.

## Width Scaling: Fan-Recurrent Atomic NoPE

`Atomic + NoPE` recurrent 조건에서 model width를 키운 결과도 확인했다.
기본 조건은 `d_model=128`, heads 4, `d_ff=512`이고, scaling 조건은
`d_model=256`, heads 8, `d_ff=1024` 및 `d_model=512`, heads 8,
`d_ff=2048`이다. 모두 layer 1 shared block, train 2~10 swaps, online
100000 steps, final CE only 조건이다.

| Model size | ID K=N | Boundary K=N | Boundary best fixed K | OOD 20~40 K=N | OOD 20~40 best fixed K | OOD 40~80 best fixed K |
|---|---:|---:|---:|---:|---:|---:|
| d128/h4/ff512 | 1.000 / 1.000 | 0.916 / 0.978 | 0.980 / 0.994 at K=24 | 0.042 / 0.358 | 0.142 / 0.470 at K=64 | 0.004 / 0.193 at K=128 |
| d256/h8/ff1024 | 1.000 / 1.000 | 0.937 / 0.982 | 0.948 / 0.986 at K=20 | 0.060 / 0.388 | 0.082 / 0.414 at K=40 | 0.002 / 0.200 at K=128 |
| d512/h8/ff2048 | 1.000 / 1.000 | 0.994 / 0.999 | 0.994 / 0.999 at K=20 | 0.182 / 0.482 | 0.308 / 0.583 at K=40 | 0.000 / 0.214 at K=40 |

Width scaling은 ID 성능을 유지하면서 boundary와 OOD 20~40 K=N을 개선한다.
d256에서는 개선 폭이 작았지만, d512에서는 boundary 11~19가 거의 해결되고
OOD 20~40도 exact 0.182까지 오른다. 다만 OOD 40~80은 여전히 chance 수준이며,
fixed loop sweep에서는 과도한 recurrence에서 state drift가 남아 있다.

### d256 K Sweep

| Split | K=N | K=20 | K=40 | K=64 | K=128 |
|---|---:|---:|---:|---:|---:|
| ID 2~10 | 1.000 / 1.000 | 0.866 / 0.972 | 0.334 / 0.824 | 0.062 / 0.678 | 0.008 / 0.536 |
| Boundary 11~19 | 0.937 / 0.982 | 0.948 / 0.986 | 0.563 / 0.887 | 0.128 / 0.700 | 0.002 / 0.494 |
| OOD 20~40 | 0.060 / 0.388 | 0.044 / 0.315 | 0.082 / 0.414 | 0.028 / 0.362 | 0.000 / 0.281 |
| OOD 40~80 | 0.000 / 0.200 | 0.000 / 0.197 | 0.000 / 0.204 | 0.000 / 0.193 | 0.002 / 0.200 |

이 패턴은 두 가지를 시사한다.

1. Width scaling은 near-boundary의 transition composition을 조금 더 잘
   학습하게 만든다. Boundary K=N은 d128의 0.916에서 d256의 0.937로 오른다.
2. 더 큰 모델은 over-computation에 더 취약하다. ID조차 K=N에서는 1.0이지만
   K=40에서 0.334, K=64에서 0.062, K=128에서 0.008까지 떨어진다.

따라서 d256 결과만 보면 "width는 학습 범위 근처의 margin을 약간 키우지만,
긴 recurrent rollout의 state drift 문제를 해결하지 못한다"에 가까웠다. d512
결과를 포함하면 결론은 더 강한 scaling-positive 쪽으로 움직인다. width는
near-boundary뿐 아니라 OOD 20~40 초반의 transition composition도 뚜렷하게
개선한다. 하지만 40~80 swaps까지의 systematic extrapolation은 아직 나오지
않는다.

### d256 Boundary Breakdown

| Swaps | K=N | K=20 | K=40 | K=64 | K=128 |
|---:|---:|---:|---:|---:|---:|
| 11 | 1.00 | 0.97 | 0.47 | 0.08 | 0.00 |
| 12 | 1.00 | 0.99 | 0.49 | 0.25 | 0.00 |
| 13 | 1.00 | 0.99 | 0.60 | 0.12 | 0.00 |
| 14 | 1.00 | 1.00 | 0.56 | 0.10 | 0.01 |
| 15 | 0.98 | 0.99 | 0.55 | 0.14 | 0.00 |
| 16 | 0.96 | 0.98 | 0.53 | 0.06 | 0.00 |
| 17 | 0.90 | 0.95 | 0.64 | 0.09 | 0.01 |
| 18 | 0.89 | 0.93 | 0.69 | 0.19 | 0.00 |
| 19 | 0.70 | 0.73 | 0.54 | 0.12 | 0.00 |

Boundary에서는 d256이 K=N과 K=20에서 안정적이지만, K=40부터는 반복 횟수가
정답 길이보다 커질수록 drift가 누적된다. 이 결과는 adaptive computation을
논할 때도 중요하다. 단순히 K를 크게 주는 방식은 도움이 되지 않고, 적정
halt 지점을 맞추는 문제가 별도로 남는다.

### d512 K Sweep

| Split | K=N | K=20 | K=40 | K=64 | K=128 |
|---|---:|---:|---:|---:|---:|
| ID 2~10 | 1.000 / 1.000 | 0.888 / 0.976 | 0.530 / 0.848 | 0.124 / 0.636 | 0.000 / 0.376 |
| Boundary 11~19 | 0.994 / 0.999 | 0.994 / 0.999 | 0.939 / 0.988 | 0.797 / 0.954 | 0.067 / 0.636 |
| OOD 20~40 | 0.182 / 0.482 | 0.090 / 0.360 | 0.308 / 0.583 | 0.286 / 0.599 | 0.190 / 0.557 |
| OOD 40~80 | 0.000 / 0.206 | 0.000 / 0.211 | 0.000 / 0.214 | 0.000 / 0.203 | 0.000 / 0.214 |

d512에서는 OOD 20~40에서 K=N보다 fixed K=40이 더 낫다. 이는 OOD 길이에 대해
추가 recurrence가 실제로 도움이 된다는 신호다. 반대로 ID와 boundary에서는
정답 길이보다 훨씬 큰 K를 주면 성능이 떨어지므로, recurrence는 많을수록 좋은
형태가 아니라 split과 길이에 맞는 적정 구간이 있다.

### d512 Boundary K Sweep

| Swaps | K=N | K=20 | K=40 | K=64 | K=128 |
|---:|---:|---:|---:|---:|---:|
| 11 | 1.00 | 0.99 | 0.90 | 0.57 | 0.00 |
| 12 | 1.00 | 0.98 | 0.90 | 0.66 | 0.00 |
| 13 | 1.00 | 1.00 | 0.95 | 0.76 | 0.00 |
| 14 | 1.00 | 1.00 | 0.92 | 0.85 | 0.01 |
| 15 | 1.00 | 0.99 | 0.97 | 0.84 | 0.00 |
| 16 | 1.00 | 1.00 | 0.94 | 0.84 | 0.02 |
| 17 | 1.00 | 1.00 | 0.96 | 0.86 | 0.14 |
| 18 | 0.98 | 1.00 | 0.96 | 0.86 | 0.16 |
| 19 | 0.97 | 0.99 | 0.95 | 0.93 | 0.27 |

d512는 boundary에서 d128/d256과 질적으로 다르다. K=N과 K=20은 사실상
완전 해결이고, K=40에서도 exact 0.939를 유지한다. K=64에서도 exact 0.797로
상당히 높다. 다만 K=128에서는 exact 0.067까지 떨어져, 충분히 큰 모델에서도
무제한 rollout 안정성이 자동으로 생기지는 않는다.

### d512 OOD 20~40 K=N Breakdown

| Swaps | Exact |
|---:|---:|
| 20 | 0.821 |
| 21 | 0.840 |
| 22 | 0.550 |
| 23 | 0.563 |
| 24 | 0.381 |
| 25 | 0.105 |
| 26 | 0.261 |
| 27 | 0.031 |
| 28 | 0.043 |
| 29~40 | 0.000 |

d512는 OOD 20~40 K=N에서 전체 exact 0.182 / slot 0.482를 기록했다. 이는 d128과
d256보다 명확한 개선이지만, 성능은 20~24 swaps에 집중되어 있고 29 swaps 이후에는
0으로 떨어진다. fixed K를 40으로 주면 overall exact가 0.308 / slot 0.583까지
오르며, 20~28 swaps 구간이 더 넓게 살아난다. 그래도 31 swaps 이후에는 거의
0이므로 scaling과 extra recurrence가 외삽 경계를 오른쪽으로 밀지만, 20~40 전체를
균일하게 푸는 단계에는 아직 도달하지 못했다.

## Depth and Width+Depth Scaling Follow-up

이번 script에서는 layer 2 recurrent block도 추가로 확인했다. 이 표는 모두
seed 0, `Atomic + NoPE`, train 2~10 swaps, final CE only 조건이다.

| Model size | Layers | ID K=N | Boundary K=N | Boundary best fixed K | OOD 20~40 K=N | OOD 20~40 best fixed K | OOD 40~80 K=N | OOD 40~80 best fixed K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d128/h4/ff512 | 1 | 1.000 / 1.000 | 0.916 / 0.978 | 0.980 / 0.994 at K=24 | 0.042 / 0.358 | 0.142 / 0.470 at K=64 | 0.000 / 0.191 | 0.004 / 0.193 at K=128 |
| d128/h4/ff512 | 2 | 1.000 / 1.000 | 0.962 / 0.991 | 0.986 / 0.996 at K=24 | 0.132 / 0.461 | 0.216 / 0.550 at K=40 | 0.000 / 0.199 | 0.000 / 0.202 at K=24 |
| d256/h8/ff1024 | 1 | 1.000 / 1.000 | 0.937 / 0.982 | 0.948 / 0.986 at K=20 | 0.060 / 0.388 | 0.082 / 0.414 at K=40 | 0.000 / 0.200 | 0.002 / 0.200 at K=128 |
| d256/h8/ff1024 | 2 | 1.000 / 1.000 | 0.991 / 0.998 | 0.994 / 0.998 at K=24/40 | 0.288 / 0.624 | 0.354 / 0.664 at K=40 | 0.000 / 0.220 | 0.006 / 0.220 at K=128 |
| d512/h8/ff2048 | 1 | 1.000 / 1.000 | 0.994 / 0.999 | 0.994 / 0.999 at K=20 | 0.182 / 0.482 | 0.308 / 0.583 at K=40 | 0.000 / 0.206 | 0.000 / 0.214 at K=40 |

Layer depth는 단순 width와 별개로 도움이 된다. d128 L2는 d128 L1보다 boundary
K=N이 0.916에서 0.962로, OOD 20~40 K=N이 0.042에서 0.132로 오른다. d256 L2는
이번 seed 0 조건에서 가장 강한 OOD 20~40 결과를 보인다. K=N exact 0.288,
fixed K=40 exact 0.354 / slot 0.664로, d512 L1의 fixed K=40 exact 0.308보다도
높다.

하지만 d256 L2도 OOD 40~80에서는 exact 0.000 K=N, best fixed 0.006에 그친다.
즉 depth와 width-depth scaling은 20~40 구간의 유효 외삽 범위를 넓히지만,
40~80 장거리 rollout 안정성은 해결하지 못했다.

## Basic Transformer Baseline

Basic baseline은 fixed-depth direct Transformer다. 아래 첫 표는 seed 0의
`Atomic + NoPE + causal` layer 1, 6, 10 결과다.

| Model | Layers | Input | PE | Causal | ID 2~10 | Boundary 11~19 | OOD 20~40 | OOD 40~80 |
|---|---:|---|---|---|---:|---:|---:|---:|
| Basic | 1 | Atomic | NoPE | yes | 0.098 / 0.426 | 0.001 / 0.207 | 0.000 / 0.198 | 0.000 / 0.206 |
| Basic | 6 | Atomic | NoPE | yes | 1.000 / 1.000 | 0.259 / 0.498 | 0.000 / 0.203 | 0.000 / 0.206 |
| Basic | 10 | Atomic | NoPE | yes | 1.000 / 1.000 | 0.280 / 0.533 | 0.000 / 0.200 | 0.000 / 0.200 |

Layer 6과 layer 10은 seed 1/2 반복에서도 같은 패턴을 보인다.

| Model | Seeds | ID K=N mean | Boundary K=N mean | OOD 20~40 K=N mean | OOD 40~80 K=N mean |
|---|---|---:|---:|---:|---:|
| Basic L6 | 0/1/2 | 1.000 / 1.000 | 0.248 / 0.499 | 0.000 / 0.201 | 0.000 / 0.202 |
| Basic L10 | 0/1/2 | 1.000 / 1.000 | 0.332 / 0.584 | 0.000 / 0.203 | 0.000 / 0.197 |

Layer 1 baseline은 ID도 제대로 풀지 못한다. ID split의 exact match는 2 swaps에서
0.542지만, 5 swaps부터 거의 0에 가깝고 7~10 swaps에서는 0이다. 따라서 이
조건의 OOD 실패는 길이 일반화 실패라기보다 depth 부족으로 인한 ID underfitting
결과로 해석해야 한다.

Layer 6과 layer 10은 ID 2~10을 완전히 풀지만 boundary 11~19에서 빠르게
무너지고 OOD 20~40, 40~80은 chance 수준이다. 따라서 fixed-depth direct
Transformer는 충분한 depth로 학습 범위는 풀 수 있지만, 학습 범위를 넘어서는
transition composition은 안정적으로 반복하지 못한다.

### Basic Boundary Breakdown (seed 0)

| Swaps | Basic L=1 | Basic L=6 | Basic L=10 |
|---:|---:|---:|---:|
| 11 | 0.00 | 0.99 | 1.00 |
| 12 | 0.00 | 0.90 | 0.93 |
| 13 | 0.00 | 0.42 | 0.46 |
| 14 | 0.00 | 0.02 | 0.12 |
| 15 | 0.01 | 0.00 | 0.01 |
| 16 | 0.00 | 0.00 | 0.00 |
| 17 | 0.00 | 0.00 | 0.00 |
| 18 | 0.00 | 0.00 | 0.00 |
| 19 | 0.00 | 0.00 | 0.00 |

Basic L=6과 L=10은 11~12 swaps에서는 높은 exact match를 보이지만 13 swaps부터
급격히 떨어진다. L=10은 L=6보다 14 swaps에서 약간 낫지만, 15 swaps 이후에는
차이가 거의 없다. 이는 단순히 fixed depth를 6에서 10으로 늘리는 것만으로는
긴 swap sequence에 필요한 반복 연산을 얻지 못한다는 증거다.

Fan-aligned `Atomic + NoPE` recurrent 조건과 비교하면 차이가 더 분명하다.
d128 L1 recurrent 모델은 seed 평균 기준 boundary K=N에서 0.906 exact, best
fixed K에서 0.938 exact를 기록했다. OOD 20~40도 낮지만 seed 평균 K=N 0.046
exact, best fixed K 0.092 exact로 Basic과 분리된다. 반면 Basic L=6/L=10은
ID가 1.0이어도 seed 0/1/2 모두에서 OOD 20~40 exact가 0.0이고, slot accuracy는
약 0.20에 머문다.

## 현재까지의 주장 가능 범위

현재 결과만으로 강하게 말할 수 있는 것은 다음이다.

- `Atomic + NoPE` recurrent 조건은 seed 0/1/2 모두에서 boundary 11~19로
  training length 밖 일반화를 보인다. d128 L1 평균 boundary K=N은 0.906 exact,
  best fixed K 평균은 0.938 exact다.
- 같은 atomic representation이라도 sinusoidal positional encoding을 넣으면
  boundary 일반화가 크게 약해지고 OOD 20~40은 chance로 붕괴한다.
- Basic layer 6과 layer 10은 seed 0/1/2 모두에서 ID를 완전히 풀지만 OOD
  20~40 exact는 전부 0.000이다. 따라서 recurrent `Atomic + NoPE`의
  boundary/OOD 개선은 단순히 parameter 수나 fixed depth 증가로 설명되지 않는다.
- `Atomic + NoPE` recurrent에서 width, depth, width+depth scaling은 모두
  효과가 있다. 특히 d256 L2 seed 0은 OOD 20~40 K=N 0.288 exact, fixed K=40
  0.354 exact까지 오른다.
- 하지만 OOD 40~80은 여전히 chance이고, 일부 조건에서는 K=128 over-rollout에서
  drift가 남는다. 따라서 scaling은 외삽 경계를 늘리지만 아직 완전한 systematic
  length generalization을 만들지는 못한다.

아직 확정하려면 필요한 값은 다음이다.

- `Atomic + Sinusoidal`의 `result.json`: ID 2~10, OOD 40~80 K=N
- scaling 조건(d128 L2, d256 L2, d512 L1)의 seed 1/2 반복
- `Atomic + Sinusoidal` seed 1/2 반복
