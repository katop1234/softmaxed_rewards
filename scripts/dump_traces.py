"""
Dump reasoning traces from tensorboard logs to text files.
Run periodically or after training to save traces.

Usage:
    python dump_traces.py                    # dump all
    python dump_traces.py --every 10         # only every 10 steps
    python dump_traces.py --watch            # continuous monitoring
"""
import os
import re
import html
import time
import argparse
from pathlib import Path

def get_traces_from_log(log_dir: str):
    """Extract traces from tensorboard log directory."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("tensorboard not installed")
        return {}
    
    if not os.path.exists(log_dir):
        return {}
    
    subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(f'{log_dir}/{d}')]
    if not subdirs:
        return {}
    
    latest = sorted(subdirs)[-1]
    ea = event_accumulator.EventAccumulator(f'{log_dir}/{latest}')
    ea.Reload()
    
    traces = {}
    try:
        tensors = ea.Tensors('text_0/text_summary')
        for t in tensors:
            text = t.tensor_proto.string_val[0].decode('utf-8')
            text = html.unescape(text.replace('<pre>', '').replace('</pre>', ''))
            
            # Extract problem
            nums = re.search(r'\[([^\]]+)\]', text)
            target = re.search(r'equals (\d+)', text)
            problem = f"Numbers: [{nums.group(1)}] → Target: {target.group(1)}" if nums and target else "Unknown"
            
            # Extract response
            response = ""
            if '<|im_start|>assistant' in text:
                response = text.split('<|im_start|>assistant')[-1]
                response = response.split('<|im_end|>')[0] if '<|im_end|>' in response else response
                response = response.strip()
            
            traces[t.step] = {
                'problem': problem,
                'response': response
            }
    except Exception as e:
        print(f"Error reading {log_dir}: {e}")
    
    return traces


def dump_traces(output_dir: str = "traces", every: int = 10):
    """Dump traces from all experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    experiments = [
        ('baseline', 'logs_baseline'),
        ('softmax_tau0.5', 'logs_softmax'),
        ('softmax_tau0.25', 'logs_tau0.25'),
        ('softmax_tau0.75', 'logs_tau0.75'),
        ('softmax_tau1.0', 'logs_tau1.0'),
    ]
    
    for exp_name, log_dir in experiments:
        if not os.path.exists(log_dir):
            continue
        
        traces = get_traces_from_log(log_dir)
        if not traces:
            continue
        
        # Filter to every N steps
        steps = sorted([s for s in traces.keys() if s % every == 0 or s == min(traces.keys())])
        
        # Write to file
        exp_file = output_dir / f"{exp_name}_traces.txt"
        with open(exp_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"EXPERIMENT: {exp_name}\n")
            f.write(f"{'='*80}\n\n")
            
            for step in steps:
                trace = traces[step]
                f.write(f"--- Step {step} ---\n")
                f.write(f"{trace['problem']}\n\n")
                f.write(f"{trace['response']}\n")
                f.write(f"\n{'─'*40}\n\n")
        
        print(f"[DUMP] {exp_name}: {len(steps)} traces → {exp_file}")
    
    # Also create a side-by-side comparison
    baseline_traces = get_traces_from_log('logs_baseline')
    softmax_traces = get_traces_from_log('logs_softmax')
    
    if baseline_traces and softmax_traces:
        common_steps = sorted(set(baseline_traces.keys()) & set(softmax_traces.keys()))
        common_steps = [s for s in common_steps if s % every == 0]
        
        comparison_file = output_dir / "comparison.txt"
        with open(comparison_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"SIDE-BY-SIDE: BASELINE vs SOFTMAX τ=0.5\n")
            f.write(f"{'='*80}\n\n")
            
            for step in common_steps[:20]:  # Limit to 20 examples
                f.write(f"{'#'*80}\n")
                f.write(f"STEP {step}\n")
                f.write(f"{'#'*80}\n\n")
                
                f.write(f">>> BASELINE\n")
                f.write(f"{baseline_traces[step]['problem']}\n\n")
                f.write(f"{baseline_traces[step]['response']}\n")
                f.write(f"\n{'─'*40}\n\n")
                
                f.write(f">>> SOFTMAX τ=0.5\n")
                f.write(f"{softmax_traces[step]['problem']}\n\n")
                f.write(f"{softmax_traces[step]['response']}\n")
                f.write(f"\n{'─'*40}\n\n")
        
        print(f"[DUMP] Comparison: {len(common_steps[:20])} examples → {comparison_file}")


def watch_and_dump(interval: int = 60, every: int = 10):
    """Continuously watch and dump traces."""
    print(f"[WATCH] Monitoring every {interval}s, dumping every {every} steps...")
    while True:
        dump_traces(every=every)
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=10, help="Dump every N steps")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")
    parser.add_argument("--output", type=str, default="traces", help="Output directory")
    args = parser.parse_args()
    
    if args.watch:
        watch_and_dump(args.interval, args.every)
    else:
        dump_traces(args.output, args.every)

