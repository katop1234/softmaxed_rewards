"""
Generate comparison charts from experiment logs.

Usage:
    python plot_results.py /path/to/experiment_dir
"""
import re
import sys
from pathlib import Path

def parse_log(log_path):
    """Parse train.log into metrics."""
    metrics = {'step': [], 'train_success': [], 'eval_success': [], 'entropy': [], 'response_len': []}
    
    with open(log_path) as f:
        content = f.read().replace('\r', '\n')
    
    current_eval = None
    for line in content.split('\n'):
        # Parse step line
        step_match = re.search(r'Step (\d+)', line)
        if step_match:
            step = int(step_match.group(1))
            metrics['step'].append(step)
            
            sr = re.search(r'success_rate: ([\d.]+)', line)
            metrics['train_success'].append(float(sr.group(1)) if sr else 0)
            
            ent = re.search(r'entropy: ([\d.]+)', line)
            metrics['entropy'].append(float(ent.group(1)) if ent else 0)
            
            rlen = re.search(r'mean_response_len: ([\d.]+)', line)
            metrics['response_len'].append(float(rlen.group(1)) if rlen else 0)
            
            metrics['eval_success'].append(current_eval)
            current_eval = None
        
        # Parse eval line
        eval_match = re.search(r'Eval success rate: ([\d.]+)', line)
        if eval_match:
            current_eval = float(eval_match.group(1))
    
    return metrics

def plot_experiment(exp_dir):
    """Generate comparison plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    exp_dir = Path(exp_dir)
    ablations = [d.name for d in exp_dir.iterdir() if d.is_dir() and (d / 'train.log').exists()]
    
    if not ablations:
        print(f"No ablations found in {exp_dir}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'baseline': '#666666', 'softmax': '#2ca02c'}
    
    for ablation in ablations:
        log_path = exp_dir / ablation / 'train.log'
        metrics = parse_log(log_path)
        
        if not metrics['step']:
            continue
        
        c = colors.get(ablation, 'blue')
        lw = 3 if 'softmax' in ablation else 2
        
        # Train success
        axes[0,0].plot(metrics['step'], metrics['train_success'], label=ablation, color=c, linewidth=lw)
        
        # Eval success
        eval_steps = [s for s, e in zip(metrics['step'], metrics['eval_success']) if e is not None]
        eval_vals = [e for e in metrics['eval_success'] if e is not None]
        if eval_vals:
            axes[0,1].plot(eval_steps, eval_vals, label=ablation, color=c, linewidth=lw, marker='o')
        
        # Entropy
        axes[1,0].plot(metrics['step'], metrics['entropy'], label=ablation, color=c, linewidth=lw)
        
        # Response length
        axes[1,1].plot(metrics['step'], metrics['response_len'], label=ablation, color=c, linewidth=lw)
    
    for ax, title, ylabel in [
        (axes[0,0], 'Train Success Rate', 'Success Rate'),
        (axes[0,1], 'Eval Success Rate', 'Success Rate'),
        (axes[1,0], 'Entropy', 'Entropy'),
        (axes[1,1], 'Response Length', 'Tokens'),
    ]:
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Experiment: {exp_dir.name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    out_path = exp_dir / 'comparison.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py /path/to/experiment_dir")
        sys.exit(1)
    
    plot_experiment(sys.argv[1])

