# status-tracking-experiment

YAI 2026 여름방학 기초NLP연구팀 Toy project

자연어로 서술된 5인 공 교환 상태추적 과제에서, 같은 블록을 반복하는 Looped
Transformer가 일반 Transformer보다 길이 일반화에 유리한지 비교한다.

## 연구 범위

- 인물 5명과 색 공 5개
- 출력은 5명 모두의 최종 공 색
- 난이도는 전체 공 교환 횟수로 정의
- ID는 학습 범위와 같은 swap length, OOD는 `ood_x4`, `ood_x8`
- 비교 모델은 Direct, Explicit CoT, Recurrent
- 위치 표현은 Sinusoidal/RoPE 중 선택
- Recurrent update는 `h = e + block(h)`
- adaptive stopping은 연속 출력 분포의 symmetric KL만 사용

세부 모델 설계와 실행 기준은 [ORIGINAL_PLAN.md](ORIGINAL_PLAN.md), 작업 범위와
인수인계 내용은 [MODEL_DESIGN_HANDOFF.md](MODEL_DESIGN_HANDOFF.md), 전체 구현 목록은
[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)를 참고한다. 데이터 형식은
[docs/DATASET.md](docs/DATASET.md)에 정리되어 있다.

## 구조

```text
data/               train / id_test / ood_x4 / ood_x8 JSONL
configs/            Basic, Looped, 전체 비교 설정
scripts/            단일·다중 seed 실행, 결과 집계와 그래프
src/data/           vocabulary, collate, dataset, 데이터 검증
src/model/          Basic/Looped 진입점과 shared classifier
src/original/       Direct, Explicit CoT, Recurrent 및 학습·평가 구현
src/trainer.py      YAML config를 실험 실행기로 연결
tests/              모델, mask, CoT, KL halting, 결과 집계 테스트
```

## 설치와 검증

```bash
python -m pip install -e '.[dev]'
python -m src.data.verify
pytest
```

## 실행

단일 모델의 빠른 실행 확인:

```bash
python scripts/run_original_experiments.py --architecture direct --smoke --device cpu
python scripts/run_original_experiments.py --architecture cot --smoke --device cpu
python scripts/run_original_experiments.py --architecture recurrent --smoke --device cpu \
  --adaptive-kl-eval
```

YAML config 진입점:

```bash
python main.py --config configs/basic_model.yaml --smoke --device cpu
python main.py --config configs/looped_model.yaml --smoke --device cpu
```

3개 seed의 Basic/Looped 비교와 swap-length별 집계:

```bash
python scripts/run_original_multiseed.py \
  --seeds 0 1 2 \
  --architectures direct recurrent \
  --position-encoding sinusoidal \
  --adaptive-kl-eval

python scripts/plot_original_results.py \
  runs/original/aggregate/summary.csv \
  --output-dir runs/original/figures
```

각 run은 ID/OOD별 slot accuracy, 5-person exact match, swap 횟수별 exact match와
checkpoint를 기록한다. KL halting을 사용하면 평균 loop 수, halt rate, 마지막
symmetric KL도 함께 저장한다.
