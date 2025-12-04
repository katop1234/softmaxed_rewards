# Experiment: countdown
    
Started: 20251204_190526
Ablations: baseline, softmax
Data: Countdown-Tasks-3to4

## Tracking Progress

```bash
# Check logs
tail -f ../results_grpo/countdown_0.5B_baseline_vs_softmax_20251204_190526/*/train.log

# Check metrics
cat ../results_grpo/countdown_0.5B_baseline_vs_softmax_20251204_190526/*/train.log | grep -E "Step|Eval"

# View tensorboard
tensorboard --logdir ../results_grpo/countdown_0.5B_baseline_vs_softmax_20251204_190526
```

## Results

Each ablation folder contains:
- `config.yaml` - Configuration used
- `train.log` - Full training log
- `logs/` - Tensorboard logs
- `checkpoints/` - Model checkpoints
- `traces/` - Sample reasoning traces
