# 데이터 파이프라인 검증. 데이터 만들거나 collate.py 고쳤으면 이거부터 돌릴 것.
#
# 핵심 아이디어: collate가 만든 토큰 시퀀스를 "토큰만 보고" 다시 파싱해서
# 교환을 처음부터 재시뮬레이션한 다음, 저장된 labels랑 맞는지 대조한다.
# 생성기가 정답을 잘못 계산했으면 여기서 걸린다.
#
#   python verify.py    (torch 필요)

import random

from data.vocab import NAMES, COLORS, SLOTS, ID2TOK, PAD, PAD_ID, TOPIC, decode
from data.data import GenConfig, sample_problem, to_row, josa
from data.collate import collate_fn, SLOT_IDS

NAME_SET = set(NAMES)


def parse_and_simulate(ids, mask):
    # 토큰 시퀀스만 보고 상태를 복원. 생성기 내부 로직과는 완전히 독립적인 경로
    toks = [ID2TOK[int(i)] for i in ids]
    state = {}
    swap_started = False
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == PAD:
            assert mask[i] == 0, f"pos {i}: PAD인데 mask가 1"
            i += 1
            continue
        if t in SLOTS:
            i += 1
            continue
        assert t in NAME_SET, f"pos {i}: 문장 시작이 이름이 아님 ({t})"

        if toks[i + 1] == TOPIC:
            # 초기 배정 문장. 교환 나온 다음에 또 나오면 뭔가 잘못된 것
            assert not swap_started
            assert toks[i] not in state, f"{toks[i]} 배정 중복"
            state[toks[i]] = toks[i + 2]
            i += 6
        else:
            swap_started = True
            a, b = toks[i], toks[i + 2]
            state[a], state[b] = state[b], state[a]
            i += 7
    return state


def check_batch(cfg, seed, n=64):
    rng = random.Random(seed)
    problems = [sample_problem(rng, cfg) for _ in range(n)]
    rows = [to_row(p, cfg) for p in problems]
    batch = collate_fn(rows)

    for i, (p, row) in enumerate(zip(problems, rows)):
        ids = batch["input_ids"][i].tolist()
        mask = batch["attn_mask"][i].tolist()
        labels = batch["labels"][i].tolist()

        # mask 규약: PAD <=> mask 0
        assert all(m == 0 for t, m in zip(ids, mask) if t == PAD_ID)
        assert all(t == PAD_ID for t, m in zip(ids, mask) if m == 0)

        # slot_pos가 진짜 SLOT 토큰을 가리키는지
        for j, pos in enumerate(batch["slot_pos"][i].tolist()):
            assert ids[pos] == SLOT_IDS[j]

        # 독립 재시뮬레이션 결과와 labels 대조
        final = parse_and_simulate(ids, mask)
        for j in range(len(SLOTS)):
            if j < cfg.n_entities:
                want = COLORS.index(final[NAMES[j]])
            else:
                want = -100
            assert (
                labels[j] == want
            ), f"sample {i} slot {j}: labels={labels[j]} 재계산={want}\n" + decode(
                ids, strip_pad=True
            )

        # text 필드가 구조 데이터랑 맞는 순서로 나오는지
        pos = -1
        frags = [
            f"{NAMES[j]}{josa(NAMES[j], '은', '는')} "
            f"{COLORS[p['init_state'][j]]} 공을"
            for j in range(cfg.n_entities)
        ]
        frags += [
            f"{NAMES[a]}{josa(NAMES[a], '과', '와')} "
            f"{NAMES[b]}{josa(NAMES[b], '이', '가')} 공을 교환했다"
            for a, b in p["swaps"]
        ]
        for fr in frags:
            nxt = row["text"].find(fr, pos + 1)
            assert nxt > pos, f"text에서 못 찾음: {fr}"
            pos = nxt


def main():
    configs = [
        (GenConfig(min_swaps=2, max_swaps=10), "기본"),
        (GenConfig(min_swaps=40, max_swaps=80, seed=7), "ood 길이"),
        (GenConfig(n_entities=3, min_swaps=2, max_swaps=10, seed=1), "인물 3명"),
        (GenConfig(min_swaps=2, max_swaps=10, noop_ratio=0.3, seed=2), "noop 노이즈"),
    ]
    for cfg, name in configs:
        for rep in range(5):
            check_batch(cfg, seed=rep * 31 + 1)
        print(f"  OK  {name:10s} (5 batches x 64)")

    # noop만 있으면 초기 상태 그대로여야 정상
    cfg = GenConfig(min_swaps=1, max_swaps=1, noop_ratio=0.999, seed=11)
    rng = random.Random(0)
    for _ in range(200):
        pr = sample_problem(rng, cfg)
        if all(a == b for a, b in pr["swaps"]):
            assert pr["init_state"] == pr["final_state"]
    print("  OK  noop이 상태 안 바꿈")

    # 라벨이 한 색으로 쏠리면 모델이 찍기를 배우니까 분포 확인
    cfg = GenConfig()
    rng = random.Random(0)
    hist = [0] * len(COLORS)
    for _ in range(20_000):
        for c in sample_problem(rng, cfg)["final_state"]:
            hist[c] += 1
    assert max(hist) / min(hist) < 1.1, f"라벨 쏠림: {hist}"
    print(f"  OK  라벨 균형 {hist}")

    print("\n전부 통과")


if __name__ == "__main__":
    main()
