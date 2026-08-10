# status-tracking-experiment

YAI 2026 여름방학 기초NLP연구팀 Toy project

자연어로 서술된 상태추적 과제에서, 같은 블록을 반복해서 도는(recurrent)
Transformer가 일반 Transformer보다 우위를 보이는가?

일반 Transformer는 레이어 수가 고정이라, 중간 토큰 생성 없이 한 번의 forward
pass로 답을 내야 하면 상태 갱신 횟수가 레이어 수를 넘는 순간 구조적으로 못
따라간다. 공 교환 문제로 그 붕괴 지점을 재고, recurrence가 그걸 얼마나
밀어내는지 비교한다.

데이터 형식과 학습 방법은 [DATASET.md](DATASET.md) 참고.

## 비교 대상

| 조건 | 구조 |
|---|---|
| Vanilla | TransformerBlock × L (독립 가중치) |
| Recurrent | TransformerBlock × 1을 T회 반복, 매 스텝 timestep encoding 가산 |

비교 짝은 유효 깊이 매칭(L=4 vs T=4)과 파라미터 매칭(L=1 vs T=4) 둘 다 본다.

## 구조

```
data/    train / id_test / ood_x4 / ood_x8 (jsonl, 예시 데이터셋 포함)
src/
├── vocab.py     고정 vocab 23개 (건드리지 말 것)
├── data.py      데이터 생성기
├── collate.py   Dataset + collate_fn (토큰화/패딩/SLOT 부착)
└── verify.py    파이프라인 검증
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

## 결과 규약

실험 결과는 `runs/*.json`으로만 주고받는다. 한 줄에
`{model, T또는L, seed, split, exact_match}` 이상을 기록할 것.
