# Training workflow

The canonical trainer now keeps model selection separate from the published
test splits. It deterministically reserves part of `train.jsonl` for
validation, stratified by `n_swaps`, selects `best.pt` using validation loss,
and evaluates that best checkpoint on the configured ID/OOD splits only after
training.

## Local outputs

For a run named `direct-sinusoidal-seed0`, outputs are written under:

```text
runs/original/
  direct-sinusoidal-seed0.json       final metrics
  direct-sinusoidal-seed0.pt         portable copy of best.pt
  direct-sinusoidal-seed0/
    metrics.jsonl                    one durable record per epoch
    checkpoints/
      last.pt                        latest epoch; use for resume
      best.pt                        lowest validation loss
      epoch_5.pt                     periodic snapshot
```

Every full checkpoint includes the model, optimizer, scheduler, AMP scaler,
training history, best metric, elapsed training time, Python/Torch/CUDA random
states, and the effective run configuration.

## Run and resume

The checked YAML configs run all values in `training.seeds`:

```bash
python main.py --config configs/basic_model.yaml --device cuda
python main.py --config configs/looped_model.yaml --device cuda
```

To resume an interrupted single run, keep the same run name and model/training
configuration, set `checkpointing.resume: auto`, and set `training.epochs` to
the intended total number of epochs. `auto` loads that run's
`checkpoints/last.pt`; it does not start the epoch count over.

An existing run cannot be started again accidentally. Choose one of:

- `checkpointing.resume: auto` to continue it;
- `checkpointing.overwrite: true` to explicitly replace it; or
- a different `run_name` to preserve it.

`resume` and `overwrite` cannot be enabled together. On resume,
`metrics.jsonl` is atomically reconstructed from checkpoint history before
training continues, which removes duplicate or incomplete epoch rows caused by
an interruption between checkpoint and log writes.

Direct CLI equivalent:

```bash
python scripts/run_original_experiments.py \
  --architecture direct --device cuda --epochs 30 --resume auto
```

AMP is enabled only on CUDA even when requested. DataLoader worker count,
pinned memory, persistent workers, optimizer, scheduler, validation ratio,
evaluation batch size/splits/metrics, and checkpoint frequency are all wired
from the YAML files into the canonical trainer.
