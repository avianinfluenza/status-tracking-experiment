"""Symbolic-first generation for controlled natural-language state tracking.

The simulator owns the gold state.  Natural language is only a rendering of
that state and its transitions; labels are never recovered from rendered text.
"""

from __future__ import annotations

import random
import re
import hashlib
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


OBJECTS = ("key", "ball", "book", "coin", "ring", "map", "cup", "toy")
LOCATIONS = tuple(f"room_{chr(ord('A') + index)}" for index in range(16))
OOD_OBJECTS = ("token", "orb", "volume", "medallion", "band", "chart", "vessel", "figurine")
OOD_LOCATIONS = tuple(f"zone_{chr(ord('A') + index)}" for index in range(16))
ACTORS = ("John", "Mary", "David", "Alice", "Jin", "Sara")
GENERATOR_VERSION = "systematic-v2"
ATOMIC_SERIALIZATION_VERSION = "systematic-atomic-v1"

INITIAL_TEMPLATES = (
    "The {entity} is in {location}.",
    "Initially, the {entity} is located in {location}.",
)
TRAIN_EVENT_TEMPLATES = (
    "{actor} moved the {entity} from {src} to {dst}.",
    "{actor} carried the {entity} from {src} into {dst}.",
)
OOD_EVENT_TEMPLATES = (
    "The {entity} was transferred from {src} to {dst} by {actor}.",
    "After taking the {entity} from {src}, {actor} placed it in {dst}.",
)
QUESTION_TEMPLATE = "Question: Where is the {entity}?"
TOKEN_PATTERN = re.compile(r"[A-Za-z]+_[A-P]|[A-Za-z]+|[?.:,]")


@dataclass(frozen=True)
class Transition:
    entity: int
    src: int
    dst: int


@dataclass(frozen=True)
class StateTrackingExample:
    text: str
    target: int
    answer: int
    target_depth: int
    num_distractors: int
    num_entities: int
    total_events: int
    events: tuple[Transition, ...]
    target_trajectory: tuple[int, ...]
    template_split: str
    initial_state: tuple[int, ...]
    target_name: str
    answer_name: str
    trajectory_names: tuple[str, ...]
    template_ids: tuple[str, ...]

    @property
    def example_id(self) -> str:
        payload = (
            self.initial_state,
            tuple((event.entity, event.src, event.dst) for event in self.events),
            self.target,
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:20]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "example_id": self.example_id,
            "generator_version": GENERATOR_VERSION,
            "target": self.target_name,
            "answer": self.answer_name,
            "label": self.answer,
            "target_depth": self.target_depth,
            "num_distractors": self.num_distractors,
            "num_entities": self.num_entities,
            "total_events": self.total_events,
            "events": [transition.__dict__ for transition in self.events],
            "target_trajectory": list(self.trajectory_names),
            "template_split": self.template_split,
            "template_ids": list(self.template_ids),
            "initial_state": list(self.initial_state),
        }


def apply_event(state: Sequence[int], event: Transition) -> list[int]:
    """Apply a transition without mutating the incoming symbolic state."""

    if state[event.entity] != event.src:
        raise ValueError("event source does not match the symbolic state")
    updated = list(state)
    updated[event.entity] = event.dst
    return updated


class StateTrackingGenerator:
    """Generate samples with independently controlled reasoning and noise."""

    def __init__(
        self,
        *,
        num_locations: int = 8,
        seed: int = 0,
        entity_names: Sequence[str] = OBJECTS,
        location_names: Sequence[str] = LOCATIONS,
    ) -> None:
        if not 2 <= num_locations <= len(LOCATIONS):
            raise ValueError("num_locations must be in [2, 16]")
        self.num_locations = num_locations
        self.rng = random.Random(seed)
        self.seed = seed
        self.entity_names = tuple(entity_names)
        self.location_names = tuple(location_names)
        if len(self.entity_names) < 2 or len(self.location_names) < num_locations:
            raise ValueError("lexical pools do not cover the requested entities/locations")

    def _different_location(self, current: int) -> int:
        candidate = self.rng.randrange(self.num_locations - 1)
        return candidate + (candidate >= current)

    def generate(
        self,
        *,
        target_depth: int,
        num_distractors: int,
        num_entities: int = 5,
        linguistic_variation: bool = False,
        template_split: str = "train",
        target: int | None = None,
    ) -> StateTrackingExample:
        if target_depth < 0 or num_distractors < 0:
            raise ValueError("depth and distractors must be non-negative")
        if not 2 <= num_entities <= len(self.entity_names):
            raise ValueError(f"num_entities must be in [2, {len(self.entity_names)}]")
        if template_split not in ("train", "ood"):
            raise ValueError("template_split must be 'train' or 'ood'")

        target = self.rng.randrange(num_entities) if target is None else target
        if not 0 <= target < num_entities:
            raise ValueError("target is outside the active entity set")
        initial_state = [self.rng.randrange(self.num_locations) for _ in range(num_entities)]
        state = list(initial_state)
        target_events: list[Transition] = []
        trajectory = [state[target]]
        for _ in range(target_depth):
            dst = self._different_location(state[target])
            target_events.append(Transition(target, state[target], dst))
            state = apply_event(state, target_events[-1])
            trajectory.append(state[target])

        distractor_events: list[Transition] = []
        # Distractors get valid sources even after random interleaving by first
        # constructing per-entity chains and preserving each chain's order.
        distractor_states = list(initial_state)
        non_targets = [index for index in range(num_entities) if index != target]
        for _ in range(num_distractors):
            entity = self.rng.choice(non_targets)
            dst = self._different_location(distractor_states[entity])
            distractor_events.append(Transition(entity, distractor_states[entity], dst))
            distractor_states[entity] = dst

        streams: dict[int, list[Transition]] = {index: [] for index in range(num_entities)}
        streams[target].extend(target_events)
        for event in distractor_events:
            streams[event.entity].append(event)
        events: list[Transition] = []
        available = [index for index, stream in streams.items() if stream]
        while available:
            entity = self.rng.choice(available)
            events.append(streams[entity].pop(0))
            if not streams[entity]:
                available.remove(entity)

        replay = list(initial_state)
        replay_trajectory = [replay[target]]
        for event in events:
            replay = apply_event(replay, event)
            if event.entity == target:
                replay_trajectory.append(replay[target])
        if tuple(replay_trajectory) != tuple(trajectory):
            raise AssertionError("target trajectory changed during interleaving")

        event_templates = TRAIN_EVENT_TEMPLATES if template_split == "train" else OOD_EVENT_TEMPLATES
        initial_templates = INITIAL_TEMPLATES if linguistic_variation else INITIAL_TEMPLATES[:1]
        if not linguistic_variation:
            event_templates = event_templates[:1]
        sentences = []
        template_ids = []
        for index in range(num_entities):
            template_index = self.rng.randrange(len(initial_templates))
            sentences.append(initial_templates[template_index].format(
                entity=self.entity_names[index], location=self.location_names[initial_state[index]]
            ))
            template_ids.append(f"initial_{template_index}")
        template_prefix = "train" if template_split == "train" else "ood"
        for event in events:
            template_index = self.rng.randrange(len(event_templates))
            sentences.append(event_templates[template_index].format(
                actor=self.rng.choice(ACTORS),
                entity=self.entity_names[event.entity],
                src=self.location_names[event.src],
                dst=self.location_names[event.dst],
            ))
            template_ids.append(f"{template_prefix}_event_{template_index}")
        sentences.append(QUESTION_TEMPLATE.format(entity=self.entity_names[target]))
        template_ids.append("question")
        return StateTrackingExample(
            text=" ".join(sentences),
            target=target,
            answer=replay[target],
            target_depth=target_depth,
            num_distractors=num_distractors,
            num_entities=num_entities,
            total_events=len(events),
            events=tuple(events),
            target_trajectory=tuple(trajectory),
            template_split=template_split,
            initial_state=tuple(initial_state),
            target_name=self.entity_names[target],
            answer_name=self.location_names[replay[target]],
            trajectory_names=tuple(self.location_names[index] for index in trajectory),
            template_ids=tuple(template_ids),
        )

    def generate_many(self, count: int, **kwargs: object) -> list[StateTrackingExample]:
        return [self.generate(**kwargs) for _ in range(count)]

    def generate_unique(
        self,
        count: int,
        *,
        seen: set[str] | None = None,
        **kwargs: object,
    ) -> list[StateTrackingExample]:
        """Generate a split with no symbolic duplicates within/across splits."""

        seen = set() if seen is None else seen
        examples: list[StateTrackingExample] = []
        attempts = 0
        while len(examples) < count:
            attempts += 1
            if attempts > max(10_000, count * 100):
                raise RuntimeError("unable to generate enough unique symbolic examples")
            example = self.generate(**kwargs)
            if example.example_id in seen:
                continue
            seen.add(example.example_id)
            examples.append(example)
        return examples


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class TokenVocabulary:
    PAD = "[PAD]"
    UNK = "[UNK]"
    CLS = "[CLS]"

    def __init__(self, tokens: Iterable[str]) -> None:
        ordered = [self.PAD, self.UNK, self.CLS]
        ordered.extend(sorted(set(tokens) - set(ordered)))
        self.itos = tuple(ordered)
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @classmethod
    def from_examples(cls, examples: Iterable[StateTrackingExample]) -> "TokenVocabulary":
        return cls(token for example in examples for token in tokenize(example.text))

    @classmethod
    def from_schema(
        cls,
        *,
        num_locations: int = 8,
        entity_names: Sequence[str] = (*OBJECTS, *OOD_OBJECTS),
        location_names: Sequence[str] = (*LOCATIONS, *OOD_LOCATIONS),
    ) -> "TokenVocabulary":
        """Build a closed vocabulary without sampling or using evaluation labels."""

        rendered = []
        for entity in entity_names:
            for location in location_names:
                rendered.extend(
                    template.format(entity=entity, location=location)
                    for template in INITIAL_TEMPLATES
                )
            rendered.append(QUESTION_TEMPLATE.format(entity=entity))
            for actor in ACTORS:
                for template in (*TRAIN_EVENT_TEMPLATES, *OOD_EVENT_TEMPLATES):
                    rendered.append(template.format(
                        actor=actor,
                        entity=entity,
                        src=location_names[0],
                        dst=location_names[1],
                    ))
        return cls(token for text in rendered for token in tokenize(text))

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def cls_id(self) -> int:
        return self.stoi[self.CLS]

    def encode(self, text: str) -> list[int]:
        return [self.cls_id] + [self.stoi.get(token, self.stoi[self.UNK]) for token in tokenize(text)]

    def encode_example(self, example: StateTrackingExample) -> list[int]:
        return self.encode(example.text)

    def __len__(self) -> int:
        return len(self.itos)


class AtomicVocabulary:
    """One token per symbolic assignment, transition, and query.

    Entity and location indices come directly from ``StateTrackingExample``;
    rendered names and templates never enter this representation.
    """

    PAD = "[PAD]"

    def __init__(self, *, num_entities: int, num_locations: int) -> None:
        if num_entities < 2 or num_locations < 2:
            raise ValueError("atomic vocabulary requires at least two entities and locations")
        self.num_entities = num_entities
        self.num_locations = num_locations
        self._init_offset = 1
        self._move_offset = self._init_offset + num_entities * num_locations
        self._query_offset = (
            self._move_offset + num_entities * num_locations * num_locations
        )
        tokens = [self.PAD]
        tokens.extend(
            f"[INIT_{entity}_{location}]"
            for entity in range(num_entities)
            for location in range(num_locations)
        )
        tokens.extend(
            f"[MOVE_{entity}_{src}_{dst}]"
            for entity in range(num_entities)
            for src in range(num_locations)
            for dst in range(num_locations)
        )
        tokens.extend(f"[QUERY_{entity}]" for entity in range(num_entities))
        self.itos = tuple(tokens)
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @property
    def pad_id(self) -> int:
        return 0

    def _validate_entity(self, entity: int) -> None:
        if not 0 <= entity < self.num_entities:
            raise ValueError("atomic example contains an out-of-range entity")

    def _validate_location(self, location: int) -> None:
        if not 0 <= location < self.num_locations:
            raise ValueError("atomic example contains an out-of-range location")

    def init_token(self, entity: int, location: int) -> int:
        self._validate_entity(entity)
        self._validate_location(location)
        return self._init_offset + entity * self.num_locations + location

    def move_token(self, entity: int, src: int, dst: int) -> int:
        self._validate_entity(entity)
        self._validate_location(src)
        self._validate_location(dst)
        return (
            self._move_offset
            + entity * self.num_locations * self.num_locations
            + src * self.num_locations
            + dst
        )

    def query_token(self, entity: int) -> int:
        self._validate_entity(entity)
        return self._query_offset + entity

    def encode_example(self, example: StateTrackingExample) -> list[int]:
        if example.num_entities > self.num_entities:
            raise ValueError("atomic vocabulary does not cover all example entities")
        tokens = [
            self.init_token(entity, location)
            for entity, location in enumerate(example.initial_state)
        ]
        tokens.extend(
            self.move_token(event.entity, event.src, event.dst)
            for event in example.events
        )
        tokens.append(self.query_token(example.target))
        return tokens

    def __len__(self) -> int:
        return len(self.itos)


Vocabulary = TokenVocabulary | AtomicVocabulary


def encode_example_row(example: StateTrackingExample, vocab: Vocabulary) -> dict[str, object]:
    return {
        "input_ids": torch.tensor(vocab.encode_example(example), dtype=torch.long),
        "example_id": example.example_id,
        "label": example.answer,
        "target_depth": example.target_depth,
        "num_distractors": example.num_distractors,
        "total_events": example.total_events,
        "trajectory": example.target_trajectory,
        "template_split": example.template_split,
        "text": example.text,
    }


class StateTrackingDataset(Dataset):
    def __init__(self, examples: Sequence[StateTrackingExample], vocab: Vocabulary) -> None:
        self.examples = list(examples)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return encode_example_row(self.examples[index], self.vocab)


def collate_examples(rows: Sequence[dict[str, object]], pad_id: int = 0) -> dict[str, object]:
    max_length = max(int(row["input_ids"].shape[0]) for row in rows)  # type: ignore[union-attr]
    input_ids = torch.full((len(rows), max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.bool)
    for index, row in enumerate(rows):
        ids = row["input_ids"]
        length = int(ids.shape[0])  # type: ignore[union-attr]
        input_ids[index, :length] = ids  # type: ignore[index]
        attention_mask[index, :length] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([int(row["label"]) for row in rows], dtype=torch.long),
        "target_depth": torch.tensor([int(row["target_depth"]) for row in rows]),
        "num_distractors": torch.tensor([int(row["num_distractors"]) for row in rows]),
        "total_events": torch.tensor([int(row["total_events"]) for row in rows]),
        "trajectory": [row["trajectory"] for row in rows],
        "example_id": [row["example_id"] for row in rows],
        "template_split": [row["template_split"] for row in rows],
        "text": [row["text"] for row in rows],
    }


def _stream_seed(base_seed: int, step: int, sample: int) -> int:
    """Mix online-example coordinates into a stable 64-bit RNG seed."""

    value = (
        (base_seed & 0xFFFFFFFFFFFFFFFF)
        ^ ((step + 1) * 0x9E3779B97F4A7C15)
        ^ ((sample + 1) * 0xBF58476D1CE4E5B9)
    ) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


class DeterministicOnlineBatchStream(Iterable[dict[str, object]]):
    """Fresh, reproducible batches under a total-event curriculum.

    Every sample in a batch has the same number of events.  This mirrors the
    ball-swap online stream and gives the batch a single recurrent loop budget
    when ``events_per_loop`` is enabled.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        batch_size: int,
        seed: int,
        min_events: int,
        max_events: int,
        steps_per_length: int,
        num_entities: int,
        num_locations: int,
        max_target_depth: int,
        max_distractors: int,
        vocab: Vocabulary,
    ) -> None:
        if min(num_steps, batch_size, steps_per_length) < 1:
            raise ValueError("online steps, batch size, and curriculum steps must be positive")
        if not 1 <= min_events <= max_events:
            raise ValueError("event curriculum must satisfy 1 <= min <= max")
        if max_events > max_target_depth + max_distractors:
            raise ValueError("event curriculum exceeds target-depth and distractor support")
        if num_entities < 2 or max_target_depth < 1 or max_distractors < 0:
            raise ValueError("invalid online generation bounds")
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.seed = seed
        self.min_events = min_events
        self.max_events = max_events
        self.steps_per_length = steps_per_length
        self.num_entities = num_entities
        self.num_locations = num_locations
        self.max_target_depth = max_target_depth
        self.max_distractors = max_distractors
        self.vocab = vocab

    def __len__(self) -> int:
        return self.num_steps

    def curriculum_max_events(self, step: int) -> int:
        if not 0 <= step < self.num_steps:
            raise IndexError("online training step is out of range")
        return min(self.max_events, self.min_events + step // self.steps_per_length)

    def _batch_for_step(self, step: int) -> dict[str, object]:
        current_max = self.curriculum_max_events(step)
        length_rng = random.Random(_stream_seed(self.seed, step, 0))
        total_events = length_rng.randint(self.min_events, current_max)
        min_depth = max(1, total_events - self.max_distractors)
        max_depth = min(self.max_target_depth, total_events)
        if min_depth > max_depth:
            raise ValueError("sampled event count has no valid depth/distractor decomposition")

        rows: list[dict[str, object]] = []
        for sample_index in range(self.batch_size):
            sample_seed = _stream_seed(self.seed, step, sample_index + 1)
            choice_rng = random.Random(sample_seed)
            target_depth = choice_rng.randint(min_depth, max_depth)
            generator = StateTrackingGenerator(
                num_locations=self.num_locations,
                seed=sample_seed,
            )
            example = generator.generate(
                target_depth=target_depth,
                num_distractors=total_events - target_depth,
                num_entities=self.num_entities,
                linguistic_variation=False,
            )
            rows.append(encode_example_row(example, self.vocab))
        return collate_examples(rows, pad_id=self.vocab.pad_id)

    def __iter__(self) -> Iterator[dict[str, object]]:
        for step in range(self.num_steps):
            yield self._batch_for_step(step)
