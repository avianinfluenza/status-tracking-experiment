# status-tracking-experiment

YAI 2026 여름방학 기초NLP연구팀 Toy project

자연어로 서술된 상태추적 과제에서, 같은 블록을 반복해서 도는(recurrent)
Transformer가 일반 Transformer보다 우위를 보이는가?

일반 Transformer의 계산 깊이는 고정되어 있지만 이것이 곧 상태 갱신 횟수가
레이어 수를 넘으면 반드시 실패한다는 뜻은 아니다. 본 연구는 그 관계를 가정하지
않고, target transition depth를 통제해 recurrent computation이 실제 추가 계산
자원으로 사용되는지를 실험적으로 검증한다.

주 실험의 가설–코드 대응과 해석 기준은 [RESEARCH_DESIGN.md](RESEARCH_DESIGN.md),
기존 ball-swap 데이터 형식은 [DATASET.md](DATASET.md) 참고. 팀의 초기 합의안을
그대로 재현하는 별도 실행 경로는 [ORIGINAL_PLAN.md](ORIGINAL_PLAN.md)에 있다.
이번 모델 작업의 배경, 변경 범위, 검증 상태와 다음 작업은
[MODEL_DESIGN_HANDOFF.md](MODEL_DESIGN_HANDOFF.md)에 정리했다.

## 비교 대상

| 조건 | 구조 |
|---|---|
| Standard | TransformerBlock × L (독립 가중치) |
| R0 Recurrent | TransformerBlock × 1을 T회 반복, conditioning 없음 |
| R1 ablation | R0 + learned loop embedding |
| Partial sharing | A/B 또는 A/B/C/D block을 순환 반복 |
| Untied control | 같은 iteration API, step마다 독립 block |

비교 짝은 compute matching과 parameter matching을 별도 표로 보고한다.

## 구조

```
data/    train / id_test / ood_x4 / ood_x8 (jsonl, 예시 데이터셋 포함)
src/
├── systematic/  target depth와 distractor를 독립 통제하는 주 연구 파이프라인
├── original/    초기 팀 스코프(Direct/Explicit CoT/Recurrent) 재현 경로
├── vocab.py     고정 vocab 23개 (건드리지 말 것)
├── data.py      데이터 생성기
├── collate.py   Dataset + collate_fn (토큰화/패딩/SLOT 부착)
├── model.py     Vanilla / Recurrent Transformer + 공통 slot 분류기
├── train.py     학습, ID/OOD 평가, checkpoint/result 저장
└── verify.py    데이터 파이프라인 검증
```

`data/`에 올라가 있는 jsonl은 기본 설정(seed 0)으로 만든 예시 데이터셋이다.
그대로 학습에 써도 되고, 조건 바꿔서 새로 만들어도 된다.

## 시작하기

```bash
cd src
python verify.py                              # 파이프라인 검증
python data.py --n-train 10000 --n-test 500   # 데이터 재생성 (data/와 동일)
```

Python 3.10+, PyTorch 필요 (데이터 생성만은 torch 없이 됨).

## 모델 학습

설계 근거와 공정 비교 규칙은 [MODELING.md](MODELING.md)에 정리되어 있다.

```bash
python -m pip install -e '.[dev]'
python -m src.verify
pytest

# 유효 깊이 매칭: L=4 vs T=4
python -m src.train --model vanilla --num-layers 4 --seed 0
python -m src.train --model recurrent --recurrent-steps 4 --seed 0

# 파라미터 매칭: L=1 vs T=4
python -m src.train --model vanilla --num-layers 1 --seed 0
python -m src.train --model recurrent --recurrent-steps 4 --seed 0
```

기존 `src/train.py`는 ball-swap 회귀/보조 실험이다. 연구 질문의 주 결과에는
`scripts/run_systematic_experiments.py`를 사용한다. 이 경로는 symbolic simulator가
gold를 만들고, `target_depth`, distractor, total events를 각각 기록한다.

R0의 기본값은 `loop_conditioning=none`, `residual_scale=1.0`이며 학습 loop보다 큰
inference loop를 허용한다. Conditioning, random-loop training, residual scaling,
adaptive halting은 주 조건과 섞지 않고 각각 ablation으로 보고한다.

```bash
python scripts/run_systematic_experiments.py --smoke
python scripts/run_systematic_experiments.py --architecture standard --num-layers 6 --seed 0
python scripts/run_systematic_experiments.py --architecture recurrent --train-loops 6 --seed 0
```

확장안 채택 여부와 무관하게 원래 팀 설계를 실행할 수 있다. 이 경로는 5명 전체
상태, 총 swap length 기준 ID/OOD, Sinusoidal/RoPE 선택, `h=e+block(h)`,
출력 KL-only halting을 고정한다.

```bash
python scripts/run_original_experiments.py --architecture direct --smoke --device cpu
python scripts/run_original_experiments.py --architecture cot --smoke --device cpu
python scripts/run_original_experiments.py --architecture recurrent --smoke --device cpu \
  --adaptive-kl-eval
```

빠른 smoke test:

```bash
python -m src.train --model recurrent --recurrent-steps 2 --epochs 1 \
  --d-model 32 --n-heads 4 --dim-feedforward 64 \
  --max-train-samples 64 --max-eval-samples 16 --device cpu
```

## 결과 규약

각 run은 결과 JSON, checkpoint와 SHA-256, per-example prediction JSONL을 함께
생성한다. 여러 seed 실행 후 다음 명령으로 raw long-format CSV와 통계를 만든다.

```bash
python scripts/aggregate_systematic_results.py results/*.json \
  --out-dir results/aggregate

python -m pip install -e '.[analysis]'
python scripts/plot_systematic_results.py results/aggregate/summary.csv \
  --out-dir results/figures
```

`raw_long.csv`는 `model × seed × depth × loop × condition` 단위를 유지한다. JSON에는
optimizer, scheduler, generator version, checkpoint hash, parameter 수, FLOPs 추정,
latency, peak memory, gradient/hidden norm이 기록된다.
