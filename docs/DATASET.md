# 데이터셋

이 레포의 데이터셋은 자연어로 서술된 공 교환 과제다. 5명의 인물이 각자
색이 다른 공 하나를 가진 상태에서 시작하고, 여러 번의 쌍 교환을 한 뒤 최종
상태를 예측한다.

데이터 생성·검증·배치 구성과 `src/model/`, `src/trainer.py` 연결까지 구현되어
있다. 아래 명령은 프로젝트 루트에서 실행한다.

## 문제 정의

인물과 색의 인덱스는 다음과 같이 고정되어 있다.

| 인덱스 | 인물 | 색 |
|---:|---|---|
| 0 | 윤성 | 빨간색 |
| 1 | 성훈 | 주황색 |
| 2 | 나영 | 노란색 |
| 3 | 정주 | 초록색 |
| 4 | 용준 | 파란색 |

초기 배정은 매 샘플마다 무작위 순열이고, 교환은 임의의 서로 다른 두 인물
사이에서 일어난다. 교환 목록을 앞에서부터 실제로 적용한 결과가 정답이다.

```text
초기: 윤성=빨강, 성훈=주황, 나영=노랑, 정주=초록, 용준=파랑
사건: 윤성과 성훈이 교환했다. 나영과 윤성이 교환했다.
정답: 윤성=노랑, 성훈=빨강, 나영=주황, 정주=초록, 용준=파랑
```

## 현재 제공되는 파일

`data/`의 JSONL 파일은 첫 줄에 메타데이터를 포함하고, 그 다음 줄부터 샘플이
하나씩 온다. 현재 체크인된 데이터는 다음 설정으로 생성되어 있다.

| 파일 | 샘플 수 | 교환 횟수 | 시드 | 용도 |
|---|---:|---:|---:|---|
| `train.jsonl` | 10,000 | 2~10 | 0 | 학습 |
| `id_test.jsonl` | 500 | 2~10 | 1000 | 학습 범위 내 평가 |
| `ood_x4.jsonl` | 500 | 20~40 | 2004 | 길이 일반화 평가 |
| `ood_x8.jsonl` | 500 | 40~80 | 2008 | 더 긴 길이 일반화 평가 |

`train`과 `id_test`는 교환 횟수 범위는 같지만 시드가 다르다. 생성기는 두
스플릿에 동일한 `(init, swaps)` 조합이 들어가지 않도록 중복을 제거한다.
OOD 스플릿은 학습 범위보다 긴 교환열을 사용한다.

길이 확장 프로파일도 별도로 제공한다. 기존 `data/` 파일은 변경하지 않고
`data/extended_length/`에 저장한다.

| 파일 | 교환 횟수 | 생성 시드 |
|---|---:|---:|
| `data/extended_length/train.jsonl` | 2~32 | 0 |
| `data/extended_length/id_test.jsonl` | 2~32 | 1000 |
| `data/extended_length/ood_x4.jsonl` | 40~80 | 2004 |
| `data/extended_length/ood_x8.jsonl` | 80~160 | 2008 |

재생성 명령:

```bash
python -m src.data.data --extended-length --out data/extended_length \
  --n-train 10000 --n-test 500 --seed 0
```

학습 시 `--extended-length`를 지정하면 기본 `data/` 대신 이 디렉터리를
자동으로 사용한다.

파일 첫 줄의 예시는 다음과 같다.

```json
{"_meta":{"n_entities":5,"min_swaps":2,"max_swaps":10,"noop_ratio":0.0,"seed":0}}
```

`BallSwapDataset`은 이 줄을 `dataset.meta`로 읽고 샘플에서는 제외한다.

## 원본 샘플 형식

```json
{
  "text": "윤성은 노란색 공을 가지고 있다. ...",
  "init": [2, 1, 0, 4, 3],
  "swaps": [[2, 3], [1, 4]],
  "labels": [0, 3, 4, 1, 2],
  "n_swaps": 2
}
```

- `init[i]`: 인물 `i`가 처음 가진 색의 인덱스
- `swaps`: 교환할 인물 쌍의 목록. 저장 시 튜플이 아니라 `[a, b]` 배열이다.
- `labels[i]`: 모든 교환 후 인물 `i`가 가진 색의 인덱스
- `n_swaps`: 교환 사건 수. 길이별 평가·분석에 사용한다.
- `text`: 사람이 읽기 위한 자연어 표현. 학습 입력은 이 문자열을 직접 토큰화하지 않고 `init`과 `swaps`에서 재구성한다.

생성기의 `n_entities`를 5보다 작게 설정할 수도 있다. 이 경우 `init`, `labels`,
`text`에는 실제 인물 수만 들어가고, 배치 라벨의 남는 슬롯은 `-100`이 된다.

## 토큰화와 배치 구성

고정 어휘는 `src/data/vocab.py`에 정의되어 있으며 총 23개다. 어휘의 순서가
토큰 ID이므로 기존 데이터나 체크포인트와 함께 사용할 때 순서를 바꾸면 안 된다.

`src/data/collate.py`의 `encode_body()`는 문자열 `text`가 아니라 구조화된
`init`과 `swaps`를 다음 고정 토큰열로 바꾼다.

```text
초기 배정 1개: [이름] [은] [색] [공을] [가지고 있다] [.]   (6토큰)
교환 1개:     [이름] [과] [이름] [이] [공을] [교환했다] [.] (7토큰)
```

배치에서는 가장 긴 본문 뒤에 짧은 샘플만 `[PAD]`를 넣고, 모든 샘플의 끝에
5개의 슬롯 토큰을 붙인다.

```text
[초기 배정 ...] [교환 ...] [PAD ...] [SLOT_윤성] ... [SLOT_용준]
```

본문 길이는 `6 * n_entities + 7 * n_swaps`이고, 배치 입력 길이는
`max_body_length + 5`다. 따라서 현재 기본 데이터의 최대 길이는 학습 105,
`ood_x4` 315, `ood_x8` 595 토큰이다. 슬롯 위치는 배치마다 동일하지만, 본문
앞쪽에 패딩이 있을 수 있으므로 모델은 반드시 `attn_mask`를 사용해야 한다.

`collate_fn()`이 반환하는 배치는 다음 dict다.

| 키 | shape | 설명 |
|---|---|---|
| `input_ids` | `[B, L]` | 토큰 ID |
| `attn_mask` | `[B, L]` | 실제 토큰/슬롯은 1, PAD는 0 |
| `slot_pos` | `[B, 5]` | 5개 슬롯 토큰의 위치 |
| `labels` | `[B, 5]` | 색 ID. 사용하지 않는 슬롯은 -100 |
| `n_swaps` | `[B]` | 교환 횟수. 길이별 exact match 집계에 사용 |

## 로더 사용

```bash
python -m src.data.verify
python -m src.data.collate data/train.jsonl
```

Python 코드에서는 다음처럼 사용한다.

```python
from src.data.collate import BallSwapDataset, make_loader

dataset = BallSwapDataset("data/train.jsonl")
print(dataset.meta)
loader = make_loader("data/train.jsonl", batch_size=256, shuffle=True)
batch = next(iter(loader))
```

원본 `text`는 collate 결과에 포함되지 않지만 `n_swaps`는 배치에 보존되므로
예측 결과를 교환 횟수별로 바로 집계할 수 있다.

## 데이터 검증

검증기는 생성기와 별도의 경로로 collate된 토큰을 다시 파싱하고, 교환을
재시뮬레이션해 `labels`와 대조한다. 또한 PAD mask, 슬롯 위치, 자연어 `text`의
순서, 라벨 분포를 검사한다.

```bash
python -m src.data.verify
```

성공하면 여러 설정에 대한 `OK` 메시지와 마지막의 `전부 통과`가 출력된다.
데이터를 새로 만들었거나 `src/data/collate.py`를 수정했다면 이 검증을 먼저
실행한다.

## 데이터 재생성

기본값으로 현재 데이터와 같은 조건의 파일을 다시 만들 수 있다. 기존 파일을
덮어쓸 수 있으므로 출력 경로를 명시한다.

```bash
python -m src.data.data --out data --n-train 10000 --n-test 500
```

생성기 주요 옵션은 다음과 같다.

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `--out` | `../data` | JSONL 출력 디렉터리 |
| `--n-train` | 10000 | 학습 샘플 수 |
| `--n-test` | 500 | `id_test`와 각 OOD 샘플 수 |
| `--n-entities` | 5 | 사용할 인물 수 (2~5) |
| `--l-train` | 10 | 학습/ID 교환 횟수 상한 |
| `--ood-mult` | `4 8` | OOD 배수 목록 |
| `--noop-ratio` | 0.0 | 자기 자신과의 교환 비율. 상태는 바뀌지 않음 |
| `--seed` | 0 | 기본 생성 시드 |

예를 들어 학습 범위를 20회까지 늘리고 OOD 배수 4, 8, 16을 추가하려면 다음과
같이 실행한다.

```bash
python -m src.data.data --out data-v2 --l-train 20 --ood-mult 4 8 16
```

재생성 시 같은 출력 파일을 덮어쓰므로, 기존 데이터가 필요하면 먼저 별도
디렉터리를 `--out`으로 지정한다.
