"""
Experiment runner with proper results tracking.
Creates unique timestamped folders that are NEVER overwritten.

Usage:
    python run_experiment.py --dataset countdown --ablations baseline,softmax
    python run_experiment.py --dataset math_level4-5 --ablations baseline,softmax
    
Results saved to:
    results_grpo/<dataset>_<ablations>_<timestamp>/
"""
import os
import sys
import yaml
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

RESULTS_BASE = "../results_grpo"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(dataset: str, ablations: List[str], model: str = None) -> Path:
    """Create unique run directory. NEVER overwrites existing."""
    ablation_str = "_vs_".join(ablations[:3])  # Limit length
    model_str = model.replace("Qwen2.5-", "").replace("-Instruct", "") if model else "3B"
    run_name = f"{dataset}_{model_str}_{ablation_str}_{timestamp()}"
    run_dir = Path(RESULTS_BASE) / run_name
    
    # Ensure uniqueness (should never happen with timestamp, but be safe)
    counter = 0
    original = run_dir
    while run_dir.exists():
        counter += 1
        run_dir = Path(f"{original}_{counter}")
    
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_config(
    run_dir: Path,
    ablation: str,
    dataset: str,
    data_path: str,
    gpu_id: int,
    tau: float = None,
    model: str = "Qwen2.5-3B-Instruct",
) -> Path:
    """Create config file for an ablation."""
    ablation_dir = run_dir / ablation
    ablation_dir.mkdir(exist_ok=True)
    
    config = {
        "model": {
            "pretrained_model_path": model,
            "device": "cuda",
            "dtype": "bfloat16",
        },
        "data": {
            "path": data_path,
            "test_size": 100,
        },
        "training": {
            "random_seed": 1337,
            "max_prompt_len": 512,
            "max_gen_len": 1024,
            "batch_size": 64,
            "num_questions_per_batch": 8,
            "micro_batch_size": 2,
            "max_grad_norm": 1.0,
            "learning_rate": 1.0e-6,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "ckpt_dir": str(ablation_dir / "checkpoints"),
            "log_dir": str(ablation_dir / "logs"),
            "skip_unfinished_episodes": False,
            "ckpt_save_interval": 50,
            "eval_interval": 10,
            "memory_efficient_adamw": True,
        },
    }
    
    # Create subdirs
    (ablation_dir / "checkpoints").mkdir(exist_ok=True)
    (ablation_dir / "logs").mkdir(exist_ok=True)
    (ablation_dir / "traces").mkdir(exist_ok=True)
    
    config_path = ablation_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path, ablation_dir


def run_ablation(
    ablation: str,
    config_path: Path,
    ablation_dir: Path,
    gpu_id: int,
    dataset: str,
    tau: float = None,
):
    """Launch an ablation experiment."""
    log_file = ablation_dir / "train.log"
    
    # Determine which script to use based on ablation type
    base_script = "train_math.py" if "math" in dataset else "train.py"
    
    if ablation == "baseline":
        script = base_script
        env_vars = f"CUDA_VISIBLE_DEVICES={gpu_id}"
    elif ablation.startswith("softmax"):
        # softmax, softmax_tau0.25, softmax_tau0.5, etc.
        script = base_script.replace("train", "train_softmax")
        t = tau or 0.5
        if "_tau" in ablation:
            t = float(ablation.split("_tau")[1])
        env_vars = f"CUDA_VISIBLE_DEVICES={gpu_id} SOFTMAX_TAU={t}"
    elif ablation == "gaussian":
        # Pure Gaussian quantile (no softmax)
        script = "train_gaussian_quantile.py"
        env_vars = f"CUDA_VISIBLE_DEVICES={gpu_id}"
    elif ablation.startswith("gaussian_softmax"):
        # gaussian_softmax, gaussian_softmax_tau0.5, etc.
        script = "train_gaussian_softmax.py"
        t = tau or 0.5
        if "_tau" in ablation:
            t = float(ablation.split("_tau")[1])
        env_vars = f"CUDA_VISIBLE_DEVICES={gpu_id} SOFTMAX_TAU={t}"
    else:
        raise ValueError(f"Unknown ablation: {ablation}")
    
    cmd = f"{env_vars} nohup python {script} --config {config_path} > {log_file} 2>&1 &"
    print(f"[{ablation}] GPU {gpu_id}: {cmd}")
    os.system(cmd)
    
    return log_file


def save_run_meta(run_dir: Path, dataset: str, ablations: List[str], data_path: str):
    """Save run metadata."""
    meta = {
        "dataset": dataset,
        "ablations": ablations,
        "data_path": data_path,
        "started": timestamp(),
        "results_dir": str(run_dir),
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Also create a README
    readme = f"""# Experiment: {dataset}
    
Started: {meta['started']}
Ablations: {', '.join(ablations)}
Data: {data_path}

## Tracking Progress

```bash
# Check logs
tail -f {run_dir}/*/train.log

# Check metrics
cat {run_dir}/*/train.log | grep -E "Step|Eval"

# View tensorboard
tensorboard --logdir {run_dir}
```

## Results

Each ablation folder contains:
- `config.yaml` - Configuration used
- `train.log` - Full training log
- `logs/` - Tensorboard logs
- `checkpoints/` - Model checkpoints
- `traces/` - Sample reasoning traces
"""
    with open(run_dir / "README.md", "w") as f:
        f.write(readme)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["countdown", "math_level1-2", "math_level3", "math_level4-5", "gsm8k"],
                       help="Dataset to use")
    parser.add_argument("--ablations", type=str, default="baseline,softmax",
                       help="Comma-separated ablations (e.g., baseline,softmax)")
    parser.add_argument("--tau", type=float, default=0.5,
                       help="Softmax temperature")
    parser.add_argument("--gpus", type=str, default="0,1",
                       help="GPUs to use (comma-separated)")
    parser.add_argument("--model", type=str, default="Qwen2.5-3B-Instruct",
                       help="Model to use (e.g., Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct)")
    
    args = parser.parse_args()
    
    ablations = args.ablations.split(",")
    gpus = [int(g) for g in args.gpus.split(",")]
    
    # Data path
    if args.dataset == "countdown":
        data_path = "Countdown-Tasks-3to4"
    else:
        data_path = f"../data/{args.dataset}"
    
    # Create run directory
    run_dir = create_run_dir(args.dataset, ablations, args.model)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {run_dir.name}")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Ablations: {ablations}")
    print(f"GPUs: {gpus}")
    print(f"Results: {run_dir}")
    print(f"{'='*60}\n")
    
    # Save metadata
    save_run_meta(run_dir, args.dataset, ablations, data_path)
    
    # Launch ablations
    for i, ablation in enumerate(ablations):
        gpu_id = gpus[i % len(gpus)]
        tau = args.tau if "softmax" in ablation else None
        
        config_path, ablation_dir = create_config(
            run_dir, ablation, args.dataset, data_path, gpu_id, tau, args.model
        )
        
        log_file = run_ablation(
            ablation, config_path, ablation_dir, gpu_id, args.dataset, tau
        )
        
        print(f"[{ablation}] Started -> {log_file}")
    
    print(f"\n{'='*60}")
    print("TRACKING PROGRESS:")
    print(f"{'='*60}")
    print(f"# Watch all logs:")
    print(f"tail -f {run_dir}/*/train.log")
    print()
    print(f"# Check metrics:")
    print(f"watch -n 30 'for f in {run_dir}/*/train.log; do echo \"=== $f ===\"; cat $f | tr \"\\r\" \"\\n\" | grep -E \"Step|Eval\" | tail -3; done'")
    print()
    print(f"# Tensorboard:")
    print(f"tensorboard --logdir {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

