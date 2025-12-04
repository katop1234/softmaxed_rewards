# Experiment: countdown
    
Started: 20251204_210358
Ablations: baseline, softmax_tau0.25, softmax_tau0.5, softmax_tau0.75, softmax_tau1.0
Data: Countdown-Tasks-3to4

## Tracking Progress

```bash
# Check logs
tail -f ../results_grpo/countdown_3B_baseline_vs_softmax_tau0.25_vs_softmax_tau0.5_20251204_210358/*/train.log

# Check metrics
cat ../results_grpo/countdown_3B_baseline_vs_softmax_tau0.25_vs_softmax_tau0.5_20251204_210358/*/train.log | grep -E "Step|Eval"

# View tensorboard
tensorboard --logdir ../results_grpo/countdown_3B_baseline_vs_softmax_tau0.25_vs_softmax_tau0.5_20251204_210358
```

## Results

Each ablation folder contains:
- `config.yaml` - Configuration used
- `train.log` - Full training log
- `logs/` - Tensorboard logs
- `checkpoints/` - Model checkpoints
- `traces/` - Sample reasoning traces
