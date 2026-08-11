# 데이터셋 설명

공 교환 상태추적 데이터셋. 뭐가 들어있고 어떻게 학습에 쓰는지 정리.

## 어떤 문제인가

5명(윤성, 성훈, 나영, 정주, 용준)이 각자 다른 색 공(빨강, 주황, 노랑, 초록, 파랑)을
하나씩 들고 시작한다. 둘씩 여러 번 교환한 뒤, 각자 최종적으로 무슨 색 공을 들고
있는지 맞히는 문제.

```
초기:  윤성=빨강, 성훈=주황, 나영=노랑, 정주=초록, 용준=파랑
사건:  윤성과 성훈이 교환했다. 나영과 윤성이 교환했다.
정답:  윤성=노랑, 성훈=빨강, 나영=주황, 정주=초록, 용준=파랑
```

교환이 N번이면 상태 갱신을 N번 순서대로 따라가야 하는데, 트랜스포머는 레이어 수가
고정이라 N이 커지면 한 번의 forward pass 안에서 못 따라간다. 어디서 무너지는지,
그리고 같은 블록을 반복해서 도는 recurrent 구조가 그걸 얼마나 버텨주는지 보는 게
이 실험의 목적.

## 파일

`data/` 안에 4개. 전부 jsonl이고 한 줄이 문제 하나다.

| 파일 | 샘플 수 | 교환 횟수 | 용도 |
|---|---|---|---|
| `train.jsonl` | 10,000 | 2~10회 | 학습 |
| `id_test.jsonl` | 500 | 2~10회 | 평가. 학습과 같은 조건, 다른 시드 |
| `ood_x4.jsonl` | 500 | 20~40회 | 평가. 학습보다 훨씬 긴 교환 |
| `ood_x8.jsonl` | 500 | 40~80회 | 평가. 더 긴 버전 |

학습 때 본 적 없는 길이에서 어떻게 되는지(ood 두 개)가 실험의 핵심이다.

**각 파일의 첫 줄은 데이터가 아니라 생성 설정 기록**이다. 직접 읽을 일이 있으면
첫 줄은 건너뛸 것 (아래 로더는 알아서 처리한다).

## 샘플 구조

파일에는 문제의 원본만 들어 있다. 토큰이나 패딩 없음.

```json
{
  "text":    "윤성은 노란색 공을 가지고 있다. ... 나영과 정주가 공을 교환했다. ...",
  "init":    [2, 1, 0, 4, 3],       // 사람i가 처음 가진 색 인덱스
  "swaps":   [[2, 3], [1, 4]],      // 교환 쌍, 사건 순서대로
  "labels":  [0, 3, 4, 1, 2],       // 각 인물의 최종 공 색 (정답)
  "n_swaps": 2                      // 교환 횟수. 결과 분석할 때 씀
}
```

- 사람 인덱스: 윤성=0, 성훈=1, 나영=2, 정주=3, 용준=4
- 색 인덱스: 빨간색=0, 주황색=1, 노란색=2, 초록색=3, 파란색=4
- `labels[0]=2`면 윤성의 최종 공이 노란색이라는 뜻
- `text`는 사람 읽으라고 넣은 필드. init/swaps에서 자동으로 만들어지는 거라
  토큰 데이터와 어긋날 일은 없고, 학습에는 안 쓴다

토큰화는 학습 시점에 `src/collate.py`가 한다. 시퀀스는 이렇게 조립된다:

```
[윤성] [은] [빨간색] [공을] [가지고 있다] [.]      초기 배정 5문장 (6토큰씩)
...
[윤성] [과] [성훈] [이] [공을] [교환했다] [.]      교환 N문장 (7토큰씩)
...
[PAD] [PAD] ...                                  배치 내 최장 샘플에 맞춰 채움
[SLOT_윤성] [SLOT_성훈] [SLOT_나영] [SLOT_정주] [SLOT_용준]
```

SLOT 토큰이 맨 끝에 오도록 PAD를 그 앞에 넣는다. 이러면 SLOT 위치가 배치 안에서
전부 같아져서 hidden state 뽑기가 gather 한 번으로 끝난다. 대신 PAD가 시퀀스
중간에 끼니까 attn_mask를 모델에 꼭 넘겨야 한다.

vocab은 딱 23개고 `src/vocab.py`에 있다. **토큰 순서를 바꾸면 다 꼬이니까
건드리지 말 것.**

## 학습시키는 법

### 1. 로더 만들기

```python
import sys; sys.path.append("src")   # 프로젝트 루트에서 돌릴 때
from collate import make_loader

train_loader = make_loader("data/train.jsonl", batch_size=256, shuffle=True)
```

collate_fn이 이미 붙어 있어서 이거면 끝이다. 배치는 dict 하나로 나온다:

| 키 | shape | 내용 |
|---|---|---|
| `input_ids` | [B, L] | 토큰 ID. L은 배치마다 다름 (배치 내 최장 기준) |
| `attn_mask` | [B, L] | 1=실제 토큰, 0=PAD |
| `slot_pos` | [B, 5] | SLOT 토큰 5개의 위치 (배치 안에서는 전부 동일) |
| `labels` | [B, 5] | 각 인물의 최종 공 색. -100이면 그 칸은 무시 |
| `n_swaps` | [B] | 교환 횟수. 길이별 exact match 집계용 |

### 2. 모델이 지켜야 할 것

모델 구조는 자유인데 (그게 실험이니까) 입출력 규약 두 가지만 맞추면 된다.

1. `forward(input_ids, attn_mask)` 형태로 받고, `attn_mask`가 0인 위치는
   attention에서 빼야 한다. `nn.TransformerEncoder` 기준으로는
   `src_key_padding_mask=(attn_mask == 0)` 넘기면 된다.
2. 출력은 전체 위치의 hidden state `[B, L, D]`. 슬롯 뽑기랑 분류는 밖에서 한다.

위치 임베딩은 신경 써서 정해야 한다. 배치마다 L이 다르고, ood 파일은 학습(최대
105)보다 훨씬 길어서(최대 595), **학습형 absolute embedding을 쓰면 ood는 아예
돌릴 수가 없다.** sinusoidal이나 RoPE처럼 길이에 안 묶이는 걸 쓰거나, 최소한
ood 최대 길이만큼 미리 잡아두거나. 어느 쪽이든 vanilla랑 recurrent가 같은 걸
써야 공정한 비교가 된다.

### 3. 학습 루프

```python
import torch
import torch.nn.functional as F
from collate import make_loader

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel(vocab_size=23).to(device)        # 모델은 각자 구현
classifier = torch.nn.Linear(D_MODEL, 5).to(device)  # 5색 분류. 슬롯 5개가 공유

opt = torch.optim.AdamW(
    list(model.parameters()) + list(classifier.parameters()), lr=3e-4)

train_loader = make_loader("data/train.jsonl", batch_size=256, shuffle=True)

for epoch in range(30):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        hidden = model(batch["input_ids"], batch["attn_mask"])   # [B, L, D]

        # 슬롯 위치 hidden state만 뽑기 -> [B, 5, D]
        idx = batch["slot_pos"].unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        slots = hidden.gather(1, idx)

        logits = classifier(slots)                               # [B, 5, 5]

        # 슬롯 5개 각각 cross entropy, 합쳐서 역전파. -100 칸은 자동 무시
        loss = F.cross_entropy(logits.reshape(-1, 5),
                               batch["labels"].reshape(-1),
                               ignore_index=-100)

        opt.zero_grad()
        loss.backward()
        opt.step()
```

시작점으로 쓸만한 설정: d_model 128, head 4, FFN 512, layer 4 (vanilla 기준),
batch 256, lr 3e-4, 30 epoch. 문제가 작아서 GPU 없어도 돌아간다.
과적합 걱정은 별로 없는 과제라 dropout은 0이어도 무방하다.

### 4. 평가

```python
@torch.no_grad()
def evaluate(model, classifier, path):
    model.eval()
    loader = make_loader(path, batch_size=256)
    n_exact = n_total = 0
    by_swaps = {}   # 교환 횟수별 집계

    for batch in loader:
        # n_swaps는 텐서에 없으니 원본에서 같이 뽑아야 함. 아래 참고
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        hidden = model(batch_gpu["input_ids"], batch_gpu["attn_mask"])
        idx = batch_gpu["slot_pos"].unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        logits = classifier(hidden.gather(1, idx))
        pred = logits.argmax(-1).cpu()                    # [B, 5]

        valid = batch["labels"] != -100
        hit = ((pred == batch["labels"]) | ~valid).all(dim=1)  # 샘플별 전부 정답?
        n_exact += hit.sum().item()
        n_total += len(hit)

    return n_exact / n_total

for split in ["id_test", "ood_x4", "ood_x8"]:
    print(split, evaluate(model, classifier, f"data/{split}.jsonl"))
```

지표는 **exact match**(5칸 전부 정답인 샘플 비율)로 통일하자. 슬롯 단위 정확도는
찍어도 점수가 나와서 모델이 무너진 걸 가려버린다.

결과 그림은 교환 횟수별로 쪼개서 그리는 게 제일 잘 보인다. 현재 collate 결과에는
`n_swaps` 텐서가 포함되므로 평가 loop에서 바로 집계할 수 있다:

```python
for n_swaps, is_exact in zip(batch["n_swaps"].tolist(), hit.tolist()):
    by_swaps.setdefault(n_swaps, []).append(is_exact)
```

x축 교환 횟수, y축 exact match로 vanilla와 recurrent를 겹쳐 그리면
"어디서 무너지는가"가 그래프 하나로 정리된다. 이게 사실상 최종 발표의 메인 그림.

### 5. 비교 실험할 때 주의

- vanilla와 recurrent는 **위치 임베딩, d_model, head 수, lr, epoch을 전부 똑같이**
  두고 구조만 달라야 한다. 하나라도 다르면 결과 차이가 구조 때문인지 설정 때문인지
  말할 수 없게 된다.
- 시드 최소 3개(예: 0, 1, 2)로 돌려서 평균±편차로 보고할 것. 작은 모델은 시드빨이
  꽤 세다.
- 비교 짝은 두 가지를 다 보는 게 좋다: 유효 깊이를 맞춘 것(vanilla 4층 vs 반복 4회)과
  파라미터 수를 맞춘 것(vanilla 1층 vs 반복 4회). 전자는 "같은 계산량", 후자는
  "같은 크기" 비교라 서로 다른 걸 말해준다.

## 데이터가 보장하는 것

- 초기 색 배정은 매번 무작위 순열 → "윤성은 늘 빨강" 같은 지름길 없음
- 모든 스플릿에서 같은 문제가 두 번 안 나옴 (train/test 겹침도 없음)
- 정답은 교환을 실제로 시뮬레이션해서 계산. `src/verify.py`가 토큰 시퀀스만 다시
  파싱해서 독립적으로 재계산해 대조함
- 5색 라벨이 정확히 균등 → 다수 클래스 찍기 불가능
- 교환 쌍은 임의의 두 명 (인접 제한 없음)

## 재생성

같은 시드면 같은 데이터가 나온다. 파일 대신 명령어를 공유하면 된다.

```bash
cd src
python verify.py                              # 검증. 재생성 전후로 꼭 실행
python data.py --n-train 10000 --n-test 500   # 지금 data/와 동일
```

실험 바꿀 때 쓰는 옵션:

| 옵션 | 효과 |
|---|---|
| `--n-train 50000` | 학습 데이터 늘리기 |
| `--l-train 20` | 학습 교환 횟수 상한 변경 |
| `--ood-mult 4 8 16` | ood 스플릿 추가 |
| `--n-entities 3` | 인물 수 줄이기 |
| `--noop-ratio 0.3` | 상태 안 바뀌는 교환 문장 섞기 (노이즈) |
| `--seed 1` | 시드 변경 |

`data/`에 있는 건 기본 설정(seed 0)으로 만든 예시 데이터셋이다. 시드가 같으면
항상 같은 데이터가 나오니까, 조건 바꾼 데이터는 파일 말고 명령어로 공유하자.
