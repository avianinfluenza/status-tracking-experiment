# 공 교환 데이터 생성기.
#
# 파일에는 문제의 원본만 저장한다: 초기 배정, 교환 목록, 정답, (읽기용) text.
# 토큰화/패딩/SLOT 부착은 학습 시점에 collate.py가 한다.
#
#   python data.py --n-train 10000 --n-test 500
#
# 만들고 나면 verify.py 꼭 돌려볼 것.

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from vocab import NAMES, COLORS, N_ENTITIES


@dataclass
class GenConfig:
    n_entities: int = 5      # 인물 수 (최대 5)
    min_swaps: int = 2
    max_swaps: int = 10      # 학습이면 L_train
    noop_ratio: float = 0.0  # 자기 자신이랑 교환하는 문장 비율. 상태는 안 바뀜 (노이즈용)
    seed: int = 0

    def __post_init__(self):
        assert 2 <= self.n_entities <= N_ENTITIES
        assert 1 <= self.min_swaps <= self.max_swaps
        assert 0.0 <= self.noop_ratio < 1.0


def sample_problem(rng, cfg):
    # 문제 하나를 만들고 교환을 실제로 굴려서 정답까지 계산
    k = cfg.n_entities

    # 초기 배정은 무작위 순열. "윤성=빨강 고정" 같은 지름길을 없애기 위함
    perm = list(range(len(COLORS)))
    rng.shuffle(perm)
    init_state = perm[:k]
    state = list(init_state)

    n_swaps = rng.randint(cfg.min_swaps, cfg.max_swaps)
    swaps = []
    for _ in range(n_swaps):
        if rng.random() < cfg.noop_ratio:
            a = rng.randrange(k)
            swaps.append((a, a))  # noop. 문장은 나가지만 상태 그대로
            continue
        a, b = rng.sample(range(k), 2)
        swaps.append((a, b))
        state[a], state[b] = state[b], state[a]

    return {
        "init_state": init_state,
        "swaps": swaps,
        "final_state": state,
        "n_swaps": n_swaps,
    }


def josa(word, with_batchim, without):
    # 받침 보고 은/는, 과/와, 이/가 고르기. text 필드에만 쓴다
    code = ord(word[-1])
    has = 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0
    return with_batchim if has else without


def render_text(problem, k):
    # 읽기용 자연어. 학습에는 안 쓴다.
    # 토큰 시퀀스는 조사가 은/과/이로 고정이라 여기랑 조금 다를 수 있음
    parts = [
        f"{NAMES[i]}{josa(NAMES[i], '은', '는')} "
        f"{COLORS[problem['init_state'][i]]} 공을 가지고 있다."
        for i in range(k)
    ]
    parts += [
        f"{NAMES[a]}{josa(NAMES[a], '과', '와')} "
        f"{NAMES[b]}{josa(NAMES[b], '이', '가')} 공을 교환했다."
        for a, b in problem["swaps"]
    ]
    return " ".join(parts)


def to_row(problem, cfg):
    return {
        "text": render_text(problem, cfg.n_entities),
        "init": problem["init_state"],
        "swaps": [list(p) for p in problem["swaps"]],
        "labels": problem["final_state"],
        "n_swaps": problem["n_swaps"],
    }


def make_split(n, cfg, seen=None):
    # 같은 문제 두 번 안 나오게 (init, swaps) 기준으로 dedup.
    # seen을 넘기면 스플릿끼리도 안 겹치게 이어서 검사한다 (train/test 누수 방지)
    rng = random.Random(cfg.seed)
    out = []
    if seen is None:
        seen = set()
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > n * 200:
            raise RuntimeError(f"{len(out)}/{n}에서 조합 고갈. n 줄이거나 max_swaps 늘릴 것")
        pr = sample_problem(rng, cfg)
        key = (tuple(pr["init_state"]), tuple(map(tuple, pr["swaps"])))
        if key in seen:
            continue
        seen.add(key)
        out.append(to_row(pr, cfg))
    return out


def write_jsonl(path, rows, cfg):
    # 첫 줄은 생성 설정. 읽는 쪽에서 건너뛰어야 함
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": asdict(cfg)}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path.name:16s} {len(rows):>6d} samples")


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        meta = json.loads(f.readline())
        rows = [json.loads(l) for l in f if l.strip()]
    return meta, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="../data")
    p.add_argument("--n-train", type=int, default=10_000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--n-entities", type=int, default=5)
    p.add_argument("--l-train", type=int, default=10)
    p.add_argument("--ood-mult", type=int, nargs="+", default=[4, 8])
    p.add_argument("--noop-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    base = dict(n_entities=a.n_entities, noop_ratio=a.noop_ratio)

    # train과 id_test는 조건이 같고 시드만 다르다.
    # seen을 공유해서 두 스플릿에 같은 문제가 들어가는 걸 막는다
    seen = set()
    print(f"[train / id_test]  swaps 2~{a.l_train}")
    cfg = GenConfig(min_swaps=2, max_swaps=a.l_train, seed=a.seed, **base)
    write_jsonl(out / "train.jsonl", make_split(a.n_train, cfg, seen), cfg)

    cfg = GenConfig(min_swaps=2, max_swaps=a.l_train, seed=a.seed + 1000, **base)
    write_jsonl(out / "id_test.jsonl", make_split(a.n_test, cfg, seen), cfg)

    # ood는 학습보다 훨씬 긴 교환. 길이 일반화 시험용
    for m in a.ood_mult:
        lo, hi = a.l_train * (m // 2 or 1), a.l_train * m
        cfg = GenConfig(min_swaps=lo, max_swaps=hi, seed=a.seed + 2000 + m, **base)
        print(f"[ood x{m}]  swaps {lo}~{hi}")
        write_jsonl(out / f"ood_x{m}.jsonl", make_split(a.n_test, cfg), cfg)


if __name__ == "__main__":
    main()
