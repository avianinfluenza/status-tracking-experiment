# Dataset + collate_fn.
#
# 파일에는 원본(init, swaps, labels)만 있으니까 토큰화/패딩/SLOT 부착을
# 여기서 배치 단위로 한다. 패딩은 배치 내 최장 샘플 기준.
#
# 기본 시퀀스 구성: 초기배정 -> 교환 -> [PAD]... -> [SLOT] 5개
# slot_first 시퀀스 구성: [SLOT] 5개 -> 초기배정 -> 교환 -> [PAD]...
# 두 형식 모두 attn_mask로 PAD를 attention에서 제외한다.

import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

try:  # module-style and script-style execution 둘 다 지원
    from .atomic import ATOMIC_PAD_ID, ATOMIC_SLOT_IDS, encode_atomic_body
    from .vocab import (NAMES, COLORS, SLOTS, TOK2ID, PAD_ID,
                        TOPIC, CONJ, SUBJ, BALL, HAVE, SWAP, PERIOD)
except ImportError:  # pragma: no cover
    from atomic import ATOMIC_PAD_ID, ATOMIC_SLOT_IDS, encode_atomic_body
    from vocab import (NAMES, COLORS, SLOTS, TOK2ID, PAD_ID,
                       TOPIC, CONJ, SUBJ, BALL, HAVE, SWAP, PERIOD)

SLOT_IDS = [TOK2ID[s] for s in SLOTS]
N_SLOTS = len(SLOTS)
EVENT_WIDTH = 7


def encode_body(init, swaps):
    # 본문 토큰만 (PAD/SLOT 없이). 초기배정 6토큰 x K + 교환 7토큰 x N
    toks = []
    for i, color in enumerate(init):
        toks += [NAMES[i], TOPIC, COLORS[color], BALL, HAVE, PERIOD]
    for a, b in swaps:
        toks += [NAMES[a], CONJ, NAMES[b], SUBJ, BALL, SWAP, PERIOD]
    return [TOK2ID[t] for t in toks]


def encode_event(a, b):
    """Encode one swap sentence without any global sequence position."""
    return [TOK2ID[t] for t in [NAMES[a], CONJ, NAMES[b], SUBJ, BALL, SWAP, PERIOD]]


class BallSwapDataset(Dataset):
    def __init__(self, path):
        with Path(path).open(encoding="utf-8") as f:
            self.meta = json.loads(f.readline())  # 첫 줄은 생성 설정
            self.rows = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # 길이가 샘플마다 달라서 여기서는 텐서로 못 만든다. collate_fn이 처리
        return self.rows[i]


def collate_fn(batch, *, slot_first=False, input_format="template"):
    if input_format not in {"template", "atomic"}:
        raise ValueError("input_format must be 'template' or 'atomic'")
    encode = encode_body if input_format == "template" else encode_atomic_body
    pad_id = PAD_ID if input_format == "template" else ATOMIC_PAD_ID
    slot_ids = SLOT_IDS if input_format == "template" else list(ATOMIC_SLOT_IDS)
    bodies = [encode(r["init"], r["swaps"]) for r in batch]
    max_body = max(len(b) for b in bodies)

    input_ids, attn_mask = [], []
    for body in bodies:
        n_pad = max_body - len(body)
        if slot_first:
            # Keep output registers at logical positions 0..N_SLOTS-1.
            input_ids.append(slot_ids + body + [pad_id] * n_pad)
            attn_mask.append([1] * N_SLOTS + [1] * len(body) + [0] * n_pad)
        else:
            input_ids.append(body + [pad_id] * n_pad + slot_ids)
            attn_mask.append([1] * len(body) + [0] * n_pad + [1] * N_SLOTS)

    # SLOT registers are fixed at the beginning only in slot_first mode.
    slot_pos = list(range(N_SLOTS)) if slot_first else list(range(max_body, max_body + N_SLOTS))

    labels = []
    for r in batch:
        lb = list(r["labels"])
        lb += [-100] * (N_SLOTS - len(lb))  # 인물 5명 미만 실험이면 남는 칸은 무시
        labels.append(lb)

    max_swaps = max(int(row["n_swaps"]) for row in batch)
    initial_colors = []
    register_mask = []
    event_input_ids = []
    event_mask = []
    for row in batch:
        init = list(row["init"])
        initial_colors.append(init + [len(COLORS)] * (N_SLOTS - len(init)))
        register_mask.append([1] * len(init) + [0] * (N_SLOTS - len(init)))
        encoded_events = [encode_event(a, b) for a, b in row["swaps"]]
        encoded_events.extend([[PAD_ID] * EVENT_WIDTH] * (max_swaps - len(encoded_events)))
        event_input_ids.append(encoded_events)
        event_mask.append([1] * int(row["n_swaps"]) + [0] * (max_swaps - int(row["n_swaps"])))

    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attn_mask": torch.tensor(attn_mask, dtype=torch.long),
        "slot_pos": torch.tensor([slot_pos] * len(batch), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        # Event-wise models use fixed latent registers and exactly one local
        # seven-token swap sentence per recurrent update.
        "initial_colors": torch.tensor(initial_colors, dtype=torch.long),
        "register_mask": torch.tensor(register_mask, dtype=torch.long),
        "event_input_ids": torch.tensor(event_input_ids, dtype=torch.long),
        "event_mask": torch.tensor(event_mask, dtype=torch.long),
        # 평가 때 교환 횟수별 exact match를 바로 집계할 수 있게 보존한다.
        "n_swaps": torch.tensor([r["n_swaps"] for r in batch], dtype=torch.long),
    }
    has_trajectory = ["intermediate_states" in row for row in batch]
    if any(has_trajectory) and not all(has_trajectory):
        raise ValueError("a batch cannot mix rows with and without intermediate_states")
    if all(has_trajectory):
        trajectory_labels = []
        for row in batch:
            states = row["intermediate_states"]
            if not isinstance(states, list) or len(states) != int(row["n_swaps"]):
                raise ValueError("intermediate_states length must match n_swaps")
            padded_states = []
            for state in states:
                if not isinstance(state, list) or len(state) != len(row["labels"]):
                    raise ValueError("each intermediate state must match labels length")
                padded_states.append(list(state) + [-100] * (N_SLOTS - len(state)))
            padded_states.extend([[-100] * N_SLOTS] * (max_swaps - len(padded_states)))
            trajectory_labels.append(padded_states)
        result["trajectory_labels"] = torch.tensor(trajectory_labels, dtype=torch.long)
    return result


def make_loader(path, batch_size=256, shuffle=False, num_workers=0, slot_first=False):
    return DataLoader(BallSwapDataset(path), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers,
                      collate_fn=partial(collate_fn, slot_first=slot_first))


if __name__ == "__main__":
    # 대충 잘 나오는지 눈으로 확인
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/train.jsonl"
    b = next(iter(make_loader(path, batch_size=4)))
    for k, v in b.items():
        print(f"  {k:10s} {tuple(v.shape)}")
    print("text:", BallSwapDataset(path)[0]["text"][:80], "...")
