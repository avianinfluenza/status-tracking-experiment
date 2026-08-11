# 기존 ball-swap 모델링 및 안정화 ablation 매뉴얼

> 이 문서는 `src/model/state_tracking.py`의 기존 ball-swap 보조 과제를 설명한다. 연구 주제의
> 주 실험과 H1–H3 판정은 `RESEARCH_DESIGN.md` 및 `src/systematic/`을 기준으로 한다.

## 1. 공통 입출력

두 encoder는 완전히 같은 인터페이스를 사용한다.

```python
hidden = encoder(input_ids, attn_mask)  # [B, L, D]
```

`StateTrackingModel`은 `slot_pos`의 5개 hidden state를 gather하고 팀원의
`feature/classifier` 설계를 반영한 공유 2-layer MLP로 5개 색을 분류한다. 사람별
별도 head를 만들지 않으며 두 encoder가 완전히 같은 classifier를 사용한다.

## 2. 공통 설계

- token embedding과 sinusoidal position encoding 사용
- `d_model=128`, `n_heads=4`, FFN 512, GELU, pre-norm
- dropout 기본값 0
- `attn_mask == 0`인 PAD 위치를 모든 self-attention의 key에서 제외
- 최종 LayerNorm과 동일한 slot classifier 사용

PAD가 본문과 SLOT 사이에 들어가므로 물리적 위치 인덱스를 그대로 사용하면 같은
sample의 SLOT 위치가 batch의 다른 sample 길이에 따라 달라진다. 구현에서는
`attn_mask.cumsum()`으로 **유효 token 기준의 논리적 위치**를 계산한다. 그래서
batch 구성이 바뀌어도 같은 sample의 예측이 바뀌지 않는다.

Sinusoidal encoding은 요청된 길이만큼 동적으로 늘어나므로 train 최대 길이 105를
넘는 OOD 최대 길이 595도 별도 설정 없이 처리한다.

## 3. 비교 모델

### Vanilla Transformer

서로 다른 파라미터를 가진 `L`개 Transformer block을 순서대로 통과한다. 각 layer는
별도로 생성해 독립적으로 초기화한다.

### Recurrent Transformer

하나의 Transformer block을 `T`번 반복한다. 주 조건 R0의 기본값은 timestep
conditioning 없음, residual scale 1.0이다. 따라서 정확히 같은 update operator를
반복하며 학습 때보다 큰 inference `T`도 허용한다.

이 구조는 하나의 block이 모든 반복 단계의 역할을 맡으므로 parameter efficiency를
얻는 대신 layer specialization을 제한한다. 다음 장치는 R0와 섞지 않고 별도
ablation으로만 둔다.

- **loop-count conditioning**: 각 반복에 서로 다른 sinusoidal step signal을 입력
- **residual scaling**: 반복 map의 raw update에 `0.5` 등의 배율을 적용해 폭주를 감쇠
- **adaptive halting**: 추론 시 sample별 slot 확률의 symmetric KL, 평균 confidence,
  hidden update ratio를 함께 사용하고 patience 동안 안정적일 때만 종료
- **diagnostics**: loop별 KL, confidence, update ratio와 sample별 실행 step을 반환

KL만 작다고 멈추면 균일한 저신뢰 예측을 수렴으로 오판할 수 있으므로 confidence와
hidden update 조건을 반드시 함께 사용한다. 고정 `T` 비교는 기본 baseline으로
유지하고 adaptive halting은 `--adaptive-eval` 조건으로 별도 보고한다.

## 4. 핵심 비교

| 목적 | Vanilla | Recurrent | 해석 |
|---|---:|---:|---|
| 유효 깊이 매칭 | L=4 | T=4 | 비슷한 sequential computation에서 weight sharing 효과 |
| 파라미터 매칭 | L=1 | T=4 | 같은 모델 크기에서 추가 recurrent computation 효과 |

모든 비교에서 `d_model`, head 수, FFN 크기, 위치 인코딩, optimizer, batch size,
학습 epoch과 seed를 동일하게 둔다.

## 5. 실행

```bash
python -m pip install -e '.[dev]'
python -m src.data.verify
pytest

# 유효 깊이 매칭
python -m src.train --model vanilla --num-layers 4 --seed 0
python -m src.train --model recurrent --recurrent-steps 4 --seed 0

# 파라미터 매칭
python -m src.train --model vanilla --num-layers 1 --seed 0
python -m src.train --model recurrent --recurrent-steps 4 --seed 0
```

최소 3개 seed(`0, 1, 2`)로 반복한다. 결과는 `runs/*.json`에 split별 JSON 한 줄로
기록되고, 최고 ID exact match checkpoint는 `checkpoints/*.pt`에 저장된다.

빠른 동작 확인에는 sample 제한 옵션을 쓸 수 있다.

```bash
python -m src.train --model recurrent --recurrent-steps 2 --epochs 1 \
  --d-model 32 --n-heads 4 --dim-feedforward 64 \
  --max-train-samples 64 --max-eval-samples 16 --device cpu
```

안정화 장치와 adaptive evaluation을 포함한 실행 예시는 다음과 같다.

```bash
python -m src.train --model recurrent --recurrent-steps 8 \
  --loop-conditioning sinusoidal --residual-scale 0.5 \
  --randomize-recurrent-steps --min-recurrent-steps 2 \
  --adaptive-eval --halting-threshold 1e-3 \
  --halting-min-confidence 0.5 --halting-update-threshold 0.25
```

## 6. 지표

- Primary: 5개 slot을 모두 맞힌 exact match
- Secondary: slot accuracy, Cross Entropy
- Generalization: `id_test`, `ood_x4`, `ood_x8`
- Breakdown: 각 split의 교환 횟수별 exact match
- Efficiency: trainable parameter 수와 학습 시간
- Adaptive efficiency: 평균 실행 step과 halt rate

`runs/*.json`의 `by_swaps`를 사용하면 교환 횟수에 따른 붕괴 곡선을 바로 그릴 수
있다.
