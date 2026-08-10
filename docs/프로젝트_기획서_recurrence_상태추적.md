# Recurrence가 상태 추적에 주는 이점 검증
## — 일반 Transformer와 Looped Transformer 비교

> 현재 레포 상태를 반영한 프로젝트 기획서

## 1. 목표

자연어로 서술된 상태 추적 과제에서, 같은 Transformer 블록을 반복 실행하는
recurrent 구조가 레이어별로 독립된 일반 Transformer보다 긴 상태 갱신열을
잘 처리하는지 검증한다.

핵심 비교는 다음 두 조건이다.

| 조건 | 구조 |
|---|---|
| Basic | 독립 파라미터를 가진 Transformer layer를 고정 횟수 실행 |
| Looped | 하나의 Transformer 블록을 파라미터 공유하며 여러 번 반복 실행 |

현재 레포에는 데이터 파이프라인과 실험 로깅 기반이 구현되어 있고, 두 모델과
학습 루프는 아직 구현 전이다. 따라서 이 문서는 확정된 데이터 설계와 앞으로
구현할 모델·실험 프로토콜을 구분해 기록한다.

## 2. 연구 질문과 가설

**RQ**: 상태를 여러 번 갱신해야 하는 자연어 과제에서 Looped Transformer가
Basic Transformer보다 높은 정확도 또는 더 나은 길이 일반화 성능을 보이는가?

**가설**: 학습 범위 안에서는 두 모델의 성능이 비슷할 수 있지만, 학습 때 보지
않은 긴 교환열에서는 recurrent 구조가 더 완만하게 성능이 저하될 것이다.
이는 가설이며, 현재 레포에는 이를 뒷받침하는 학습 결과가 아직 없다.

## 3. 태스크

5명의 인물(윤성, 성훈, 나영, 정주, 용준)이 서로 다른 색 공을 하나씩 가진
상태에서 출발한다. 이후 임의의 두 인물이 공을 교환하는 사건이 순서대로
주어지고, 모든 사건이 끝난 뒤 각 인물이 가진 공의 색을 예측한다.

```text
초기: 윤성=빨강, 성훈=주황, 나영=노랑, 정주=초록, 용준=파랑
사건: 윤성과 성훈이 교환했다. 나영과 윤성이 교환했다.
정답: 윤성=노랑, 성훈=빨강, 나영=주황, 정주=초록, 용준=파랑
```

이 과제는 표면형 다양성보다 순차적인 상태 합성 능력에 집중하기 위해 직접
생성한다. 자연어 표현은 한국어 고정 템플릿을 사용하고, 학습 입력은 생성된
`text`를 다시 분석하지 않고 `init`과 `swaps`에서 토큰열을 구성한다.

## 4. 현재 데이터 설계

데이터 생성기와 검증기는 `src/data/`에 있다. 고정 vocabulary는
`src/data/vocab.py`에 정의되어 있으며 총 23개다.

| 항목 | 현재 설정 |
|---|---|
| 인물 수 | 5명 (생성기는 2~5명 지원) |
| 색 수 | 5개 |
| 초기 상태 | 색 인덱스의 무작위 순열 |
| 사건 | 임의의 서로 다른 두 인물의 pairwise swap |
| 학습/ID 길이 | 2~10회 swap |
| OOD 길이 | 20~40회, 40~80회 swap |
| supervision | 최종 상태만 |
| 데이터 형식 | 첫 줄 메타데이터 + 샘플별 JSONL |

현재 체크인된 데이터셋은 다음과 같다.

| Split | 샘플 수 | swap 범위 | 생성 시드 |
|---|---:|---:|---:|
| `train.jsonl` | 10,000 | 2~10 | 0 |
| `id_test.jsonl` | 500 | 2~10 | 1000 |
| `ood_x4.jsonl` | 500 | 20~40 | 2004 |
| `ood_x8.jsonl` | 500 | 40~80 | 2008 |

`train`과 `id_test`는 범위가 같고 시드가 다르다. 생성기는 두 split 사이의
`(init, swaps)` 중복도 제거한다. OOD split은 학습보다 긴 사건열로 길이
일반화를 측정한다.

샘플의 핵심 필드는 다음과 같다.

```json
{
  "text": "윤성은 노란색 공을 가지고 있다. ...",
  "init": [2, 1, 0, 4, 3],
  "swaps": [[2, 3], [1, 4]],
  "labels": [0, 3, 4, 1, 2],
  "n_swaps": 2
}
```

정답은 생성기가 교환을 실제로 시뮬레이션해 계산한다. `python -m data.verify`
검증기는 collate된 토큰열을 독립적으로 다시 파싱하고, 재시뮬레이션 결과와
`labels`를 대조한다.

자세한 데이터 형식, 재생성 옵션, 배치 규약은 [DATASET.md](DATASET.md)에
기록한다.

## 5. 입력과 출력 표현

`src/data/collate.py`는 각 샘플을 다음 구조로 구성한다.

```text
[초기 배정 6토큰 × 인물 수]
[교환 7토큰 × swap 수]
[PAD ...]
[SLOT_윤성] [SLOT_성훈] [SLOT_나영] [SLOT_정주] [SLOT_용준]
```

초기 배정 한 문장은 `[이름] [은] [색] [공을] [가지고 있다] [.]`, 교환 한
문장은 `[이름] [과] [이름] [이] [공을] [교환했다] [.]`다. PAD는 배치 내
최장 본문에 맞춰 슬롯 앞에만 들어간다.

`collate_fn()`의 반환값은 다음 네 가지다.

| 키 | shape | 의미 |
|---|---|---|
| `input_ids` | `[B, L]` | 토큰 ID |
| `attn_mask` | `[B, L]` | PAD는 0, 나머지는 1 |
| `slot_pos` | `[B, 5]` | 최종 상태를 읽을 슬롯 위치 |
| `labels` | `[B, 5]` | 색 ID, 미사용 슬롯은 -100 |

현재 최대 입력 길이는 train 105, `ood_x4` 315, `ood_x8` 595다. 따라서 모델
구현 시 위치 표현이 OOD 길이를 처리할 수 있어야 한다. 또한 PAD가 본문과 슬롯
사이에 있으므로 attention에 `attn_mask`를 반드시 전달해야 한다.

## 6. 모델 비교 설계

두 모델은 vocabulary, 입력 표현, hidden dimension, attention head 수, optimizer,
학습 데이터, seed를 공유하고 recurrence 유무와 실행 구조만 다르게 한다.
두 모델 모두 사전학습 없이 from-scratch로 학습한다.

### Basic Transformer

레이어마다 독립 파라미터를 가진 Transformer block을 `L`개 순차 실행한다.

### Looped Transformer

Transformer block 하나를 공유 파라미터로 `T`회 반복 실행한다. 반복 단계 정보를
구분할 timestep encoding을 사용할지는 모델 구현 시 확정하고, 사용한다면 두
모델 비교에서 공정한 기준을 문서화한다.

우선 비교할 두 매칭은 다음과 같다.

1. **유효 깊이 매칭**: Basic `L=4` 대 Looped `T=4`
2. **블록 파라미터 수 매칭**: Basic `L=1` 대 Looped `T=4`

hidden dimension, head 수, FFN 크기, dropout, 위치 표현은 두 조건에서
동일하게 둔다. 기본 후보는 `d_model=128`, head 4이며, 실제 값은 모델 구현과
parameter count 확인 후 config에 확정한다.

현재 `src/model/basic_transformer.py`, `src/model/looped_transformer.py`,
`src/trainer.py`, `configs/*.yaml`은 파일만 준비되어 있고 내용은 아직 없다.
`main.py`도 config를 읽고 override하는 진입부까지만 구현되어 있으며 trainer
호출은 연결되지 않았다.

## 7. 학습·평가 프로토콜

모델은 `[B, L, D]` 형태의 전체 hidden state를 출력하고, `slot_pos`로 슬롯
hidden state `[B, 5, D]`를 뽑는다. 슬롯 간 공유 분류기에서 5개 색의 logits를
계산하고, `labels`에 대해 cross entropy를 적용한다. `-100` 슬롯은 loss에서
무시한다.

주요 지표는 다음과 같다.

- **state/exact accuracy**: 한 샘플의 유효한 모든 슬롯을 맞춘 비율. 메인 지표
- **slot accuracy**: 슬롯 단위 정확도. 보조 지표
- **swap별 state accuracy**: `n_swaps`별 exact accuracy. 길이 일반화의 핵심 그림

최소 3개 seed를 사용해 평균과 편차를 보고한다. 각 run에는 조건과 결과를
재현할 수 있도록 다음을 남긴다.

```text
runs/<run_id>/
├── config.final.yaml
├── command.txt
├── git.txt
├── metrics.jsonl
├── best.txt             # best checkpoint가 있을 때
└── checkpoints/
    ├── last.pt
    └── best.pt           # 설정에 따라 생성
```

이 run 디렉터리와 checkpoint 정책은 이미 구현된
`src/utils/experiment.py`가 담당한다. 권장 metric 이름은
`train_loss`, `train_slot_acc`, `train_state_acc`, `valid_loss`,
`valid_slot_acc`, `valid_state_acc`이며, OOD 결과는 예를 들어
`ood_swap_20_state_acc`처럼 기록한다.

## 8. 검증 순서

모델 학습 코드가 구현되면 다음 순서로 실험한다.

1. `src/data/verify.py`로 데이터 파이프라인 검증
2. 작은 subset과 짧은 epoch으로 Basic/Looped smoke test
3. `id_test`에서 학습 범위 성능 확인
4. `ood_x4`, `ood_x8`에서 전체 상태 정확도 측정
5. `n_swaps`별 결과를 집계하고 seed별 평균·편차 계산
6. 유효 깊이 매칭과 파라미터 수 매칭을 각각 비교

실험 결과는 모델별·seed별로 분리해 저장하고, 학습 조건이 다른 결과를 하나의
곡선으로 합치지 않는다.

## 9. 현재 상태와 다음 작업

### 완료

- [x] 공 교환 데이터 생성기
- [x] train/ID/OOD JSONL 데이터 생성
- [x] 고정 23개 vocabulary
- [x] Dataset, tokenization, padding, SLOT 구성
- [x] 독립 재파싱 기반 데이터 검증기
- [x] seed/device 유틸리티와 run/checkpoint 로거 기반

### 다음 구현 작업

- [ ] Basic Transformer 구현
- [ ] Looped Transformer 구현
- [ ] 모델 출력과 슬롯 분류기 연결
- [ ] train/evaluate loop 및 `main.py` trainer 연결
- [ ] `configs/basic_model.yaml`, `configs/looped_model.yaml` 작성
- [ ] 위치 표현과 attention mask 동작 테스트
- [ ] 3개 이상 seed의 Basic/Looped 비교 실행
- [ ] swap별 결과 집계와 그래프 작성

### 이후 확장

- [ ] 반복 단계별 deep supervision
- [ ] 중간 상태를 생성하는 CoT/teacher-forcing 조건
- [ ] 표면형 다양성 또는 noop noise를 추가한 강건성 실험
