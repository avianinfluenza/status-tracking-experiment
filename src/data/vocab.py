# 고정 vocab 23개.
# 토큰 순서가 곧 ID라서, 순서 바꾸면 만들어둔 데이터/체크포인트 전부 못 쓰게 된다.
# 절대 건드리지 말 것.

NAMES = ["윤성", "성훈", "나영", "정주", "용준"]
COLORS = ["빨간색", "주황색", "노란색", "초록색", "파란색"]

TOPIC = "은"
CONJ = "과"
SUBJ = "이"
BALL = "공을"
HAVE = "가지고 있다"
SWAP = "교환했다"
PERIOD = "."
PAD = "[PAD]"

SLOTS = [f"[SLOT_{n}]" for n in NAMES]

VOCAB = NAMES + COLORS + [TOPIC, CONJ, SUBJ, BALL, HAVE, SWAP, PERIOD, PAD] + SLOTS

TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}

VOCAB_SIZE = len(VOCAB)  # 23
PAD_ID = TOK2ID[PAD]
N_ENTITIES = len(NAMES)  # 5
N_LABELS = len(COLORS)  # 분류기 출력 크기

assert VOCAB_SIZE == 23


def decode(ids, strip_pad=False):
    # 디버깅용. ID 리스트를 읽을 수 있는 문자열로
    toks = [ID2TOK[int(i)] for i in ids]
    if strip_pad:
        toks = [t for t in toks if t != PAD]
    return " ".join(toks)
