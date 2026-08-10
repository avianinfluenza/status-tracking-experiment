# Dataset + collate_fn.
#
# 파일에는 원본(init, swaps, labels)만 있으니까 토큰화/패딩/SLOT 부착을
# 여기서 배치 단위로 한다. 패딩은 배치 내 최장 샘플 기준.
#
# 시퀀스 구성: 초기배정 -> 교환 -> [PAD]... -> [SLOT] 5개
# SLOT을 맨 끝에 두면 위치가 배치 안에서 다 같아져서 gather 한 번이면 된다.
# 대신 PAD가 중간에 끼니까 attn_mask 꼭 모델에 넘길 것.

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from data.vocab import (
    NAMES,
    COLORS,
    SLOTS,
    TOK2ID,
    PAD_ID,
    TOPIC,
    CONJ,
    SUBJ,
    BALL,
    HAVE,
    SWAP,
    PERIOD,
)

SLOT_IDS = [TOK2ID[s] for s in SLOTS]
N_SLOTS = len(SLOTS)


def encode_body(init, swaps):
    # 본문 토큰만 (PAD/SLOT 없이). 초기배정 6토큰 x K + 교환 7토큰 x N
    toks = []
    for i, color in enumerate(init):
        toks += [NAMES[i], TOPIC, COLORS[color], BALL, HAVE, PERIOD]
    for a, b in swaps:
        toks += [NAMES[a], CONJ, NAMES[b], SUBJ, BALL, SWAP, PERIOD]
    return [TOK2ID[t] for t in toks]


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


def collate_fn(batch):
    bodies = [encode_body(r["init"], r["swaps"]) for r in batch]
    max_body = max(len(b) for b in bodies)

    input_ids, attn_mask = [], []
    for body in bodies:
        n_pad = max_body - len(body)
        input_ids.append(body + [PAD_ID] * n_pad + SLOT_IDS)
        attn_mask.append([1] * len(body) + [0] * n_pad + [1] * N_SLOTS)

    # SLOT은 항상 마지막 5칸이라 배치 안에서 위치가 다 같다
    slot_pos = list(range(max_body, max_body + N_SLOTS))

    labels = []
    for r in batch:
        lb = list(r["labels"])
        lb += [-100] * (N_SLOTS - len(lb))  # 인물 5명 미만 실험이면 남는 칸은 무시
        labels.append(lb)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attn_mask": torch.tensor(attn_mask, dtype=torch.long),
        "slot_pos": torch.tensor([slot_pos] * len(batch), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def make_loader(path, batch_size=256, shuffle=False, num_workers=0):
    return DataLoader(
        BallSwapDataset(path),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    # 대충 잘 나오는지 눈으로 확인
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "../data/train.jsonl"
    b = next(iter(make_loader(path, batch_size=4)))
    for k, v in b.items():
        print(f"  {k:10s} {tuple(v.shape)}")
    print("text:", BallSwapDataset(path)[0]["text"][:80], "...")
