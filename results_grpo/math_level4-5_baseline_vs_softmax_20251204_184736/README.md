# Experiment: math_level4-5
    
Started: 20251204_184736
Ablations: baseline, softmax
Data: ../data/math_level4-5

## Tracking Progress

```bash
# Check logs
tail -f ../results_grpo/math_level4-5_baseline_vs_softmax_20251204_184736/*/train.log

# Check metrics
cat ../results_grpo/math_level4-5_baseline_vs_softmax_20251204_184736/*/train.log | grep -E "Step|Eval"

# View tensorboard
tensorboard --logdir ../results_grpo/math_level4-5_baseline_vs_softmax_20251204_184736
```

## Results

Each ablation folder contains:
- `config.yaml` - Configuration used
- `train.log` - Full training log
- `logs/` - Tensorboard logs
- `checkpoints/` - Model checkpoints
- `traces/` - Sample reasoning traces
