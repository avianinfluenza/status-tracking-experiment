# 최종 연구기획서 대비 구현 점검

검토 기준: `recurrent_transformer_state_tracking_research_plan_ko.pdf` 12쪽 전체.

## 판정

초기 systematic 구현은 핵심 아이디어(H1–H5)는 반영했지만, 최종 기획서의 정확한
프로토콜과 논문 수준 재현성 요구에는 일부 미달했다. 이 보완본에서는 아래 항목을
추가해 controlled object-location 확장 실험 설계와 구현을 일치시켰다. 현재 실행
우선순위는 `src/original/` ball-swap에서 R0 recurrent 조건을 먼저 검증하는 것이다.

| 기획서 요구 | 이전 상태 | 보완 상태 |
|---|---|---|
| Train D=1–8, OOD D=10/12/16/20/24/32 | D≤6 기본, OOD 일부 누락 | 기본값과 누수 assert 반영 |
| OOD cell당 2,000–5,000 | 500 | 기본 2,000/cell |
| E0 모든 ID depth ≥95% | 전체 평균만 확인 | depth별 최솟값으로 validity gate |
| OOD degradation slope | 없음 | run마다 least-squares slope 저장 |
| K=1,2,4,6,8,12,16,24 | 일부만 | 전체 기본 sweep |
| total_events=24, D=2/4/8/12/16/20 | event 수/점 불일치 | 정확히 반영 |
| D=8, distractor 0–64 | 최대 32 | 0/4/8/16/32/64 반영 |
| template OOD와 lexical memorization 분리 | template만 | lexical alias split 추가 |
| sharing ratio ablation | 1 block/완전 untied만 | cyclic 1/2/4/T block 지원 |
| split 중복 방지와 generator version | 없음 | symbolic fingerprint dedup/version |
| checkpoint hash | 없음 | `.pt`와 SHA-256 저장 |
| per-example prediction/confidence/NLL | 없음 | depth/ID JSONL 저장 |
| FLOPs/latency/max memory | 없음 | 추정 FLOPs, 실측 latency, CUDA peak memory |
| long-format raw CSV | 없음 | 자동 집계 스크립트 추가 |
| seed 통계·CI·effect size·Holm | 없음 | bootstrap/paired test/Holm 구현 |
| Spearman ρ(D,K*) | 없음 | aggregate 통계에 추가 |
| raw CSV 기반 Figure 2–5 | 없음 | plotting script 추가 |

## 그대로 유지한 올바른 설계

- symbolic simulator가 gold state와 trajectory를 보유하고 자연어는 renderer로만 사용
- Standard와 Recurrent가 tokenizer, embedding, positional encoding, head를 공유
- R0는 loop conditioning 없는 정확한 weight-tied recurrence
- loop embedding, random-loop training, residual scaling, adaptive halting은 ablation
- final-state CE만 사용하고 intermediate supervision은 main condition에서 제외
- probe train/test는 main model train split과 분리하고 encoder를 고정
- 학습 loop보다 큰 inference K를 허용하고 overthinking 영역까지 평가
- compute-matched와 parameter-matched 결과를 별도로 저장

## 실행 및 판정 순서

1. Standard와 R0를 seed 0,1,2 이상으로 실행한다.
2. 두 모델의 모든 ID depth가 95% 기준을 넘었는지 확인한다.
3. E1 slope와 세 개 이상 OOD depth의 paired gap을 확인한다.
4. E2에서 `Spearman ρ(D,K*)>0`이고 seed별 방향이 일치하는지 확인한다.
5. E3 matched-length에서도 recurrent-standard gap이 D와 함께 증가하는지 확인한다.
6. H2+H3+H4가 함께 지지될 때만 systematic state updating의 강한 근거로 표현한다.

Smoke run의 낮은 accuracy는 64 samples/1 epoch 실행 검증 결과이며 연구 결과가 아니다.
