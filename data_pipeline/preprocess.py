"""
Data preprocessing pipeline for math reasoning datasets.

Design principles:
1. Modular: Each dataset has its own processor
2. Validated: Sanity checks at every step
3. Cached: Processed data saved to disk
4. Debuggable: Verbose logging and sample inspection

Usage:
    python preprocess.py --dataset math --level 1,2,3 --output data/math_easy
    python preprocess.py --dataset gsm8k --output data/gsm8k
    python preprocess.py --dataset aime --output data/aime
    python preprocess.py --inspect data/math_easy  # View samples
"""

import os
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MathProblem:
    """Standardized format for all math problems."""
    id: str                    # Unique identifier
    problem: str               # The question text
    answer: str                # Ground truth answer (extracted, cleaned)
    solution: Optional[str]    # Full solution text (if available)
    source: str                # Dataset source (math, gsm8k, aime, etc.)
    difficulty: Optional[int]  # 1-5 scale (if available)
    subject: Optional[str]     # algebra, geometry, etc.
    metadata: Dict[str, Any]   # Any extra info
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'MathProblem':
        return cls(**d)
    
    def validate(self) -> List[str]:
        """Return list of validation errors (empty if valid)."""
        errors = []
        if not self.problem or len(self.problem) < 10:
            errors.append(f"Problem too short: {len(self.problem)} chars")
        if not self.answer:
            errors.append("Answer is empty")
        if not self.id:
            errors.append("ID is empty")
        return errors


# =============================================================================
# Base Processor
# =============================================================================

class DatasetProcessor(ABC):
    """Base class for dataset processors."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass
    
    @abstractmethod
    def load_raw(self) -> List[dict]:
        """Load raw data from source."""
        pass
    
    @abstractmethod
    def process_item(self, raw: dict, idx: int) -> Optional[MathProblem]:
        """Convert raw item to MathProblem. Return None to skip."""
        pass
    
    def process_all(self, max_items: Optional[int] = None) -> List[MathProblem]:
        """Process entire dataset with validation."""
        logger.info(f"Loading {self.name} dataset...")
        raw_data = self.load_raw()
        logger.info(f"Loaded {len(raw_data)} raw items")
        
        if max_items:
            raw_data = raw_data[:max_items]
            logger.info(f"Limited to {max_items} items")
        
        problems = []
        skipped = 0
        validation_errors = []
        
        for idx, raw in enumerate(raw_data):
            try:
                problem = self.process_item(raw, idx)
                if problem is None:
                    skipped += 1
                    continue
                
                errors = problem.validate()
                if errors:
                    validation_errors.append((idx, errors))
                    continue
                
                problems.append(problem)
                
            except Exception as e:
                logger.warning(f"Error processing item {idx}: {e}")
                skipped += 1
        
        # Report
        logger.info(f"Processed: {len(problems)} problems")
        logger.info(f"Skipped: {skipped} items")
        if validation_errors:
            logger.warning(f"Validation errors: {len(validation_errors)}")
            for idx, errs in validation_errors[:3]:
                logger.warning(f"  Item {idx}: {errs}")
        
        return problems


# =============================================================================
# MATH Dataset Processor (AMC/AIME/Olympiad)
# =============================================================================

class MATHProcessor(DatasetProcessor):
    """Process the Hendrycks MATH dataset (lighteval/MATH)."""
    
    def __init__(self, levels: Optional[List[int]] = None, subjects: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.levels = levels  # Filter by difficulty (1-5)
        self.subjects = subjects  # Filter by subject
    
    @property
    def name(self) -> str:
        return "MATH"
    
    # All available subjects in the dataset
    ALL_SUBJECTS = [
        'algebra', 'counting_and_probability', 'geometry', 
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
    ]
    
    def load_raw(self) -> List[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")
        
        # Load all subjects or filtered subjects
        subjects_to_load = self.subjects if self.subjects else self.ALL_SUBJECTS
        
        all_data = []
        for subject in subjects_to_load:
            try:
                ds = load_dataset('EleutherAI/hendrycks_math', subject, split='train+test')
                data = [dict(item) for item in ds]
                all_data.extend(data)
                logger.info(f"  Loaded {subject}: {len(data)} items")
            except Exception as e:
                logger.warning(f"  Failed to load {subject}: {e}")
        
        logger.info(f"Total loaded: {len(all_data)} items")
        
        # Filter by level
        if self.levels:
            level_strs = [f"Level {l}" for l in self.levels]
            all_data = [d for d in all_data if d.get('level') in level_strs]
            logger.info(f"Filtered to levels {self.levels}: {len(all_data)} items")
        
        return all_data
    
    def extract_boxed_answer(self, solution: str) -> str:
        """Extract answer from \\boxed{...} in solution."""
        import re
        
        # Find all boxed expressions
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, solution)
        
        if matches:
            return matches[-1].strip()  # Return last boxed (usually the final answer)
        
        # Fallback: try simpler pattern
        simple = re.search(r'\\boxed\{([^}]+)\}', solution)
        if simple:
            return simple.group(1).strip()
        
        return ""
    
    def process_item(self, raw: dict, idx: int) -> Optional[MathProblem]:
        problem_text = raw.get('problem', '')
        solution_text = raw.get('solution', '')
        
        # Extract answer from boxed
        answer = self.extract_boxed_answer(solution_text)
        if not answer:
            logger.debug(f"No boxed answer found in item {idx}")
            return None
        
        # Parse level
        level_str = raw.get('level', '')
        level = None
        if level_str:
            try:
                level = int(level_str.replace('Level ', ''))
            except:
                pass
        
        return MathProblem(
            id=f"math_{idx}",
            problem=problem_text,
            answer=answer,
            solution=solution_text,
            source="lighteval/MATH",
            difficulty=level,
            subject=raw.get('type'),
            metadata={'original_level': level_str}
        )


# =============================================================================
# GSM8K Processor (Grade School Math)
# =============================================================================

class GSM8KProcessor(DatasetProcessor):
    """Process GSM8K dataset."""
    
    @property
    def name(self) -> str:
        return "GSM8K"
    
    def load_raw(self) -> List[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")
        
        ds = load_dataset('gsm8k', 'main', split='train+test', trust_remote_code=True)
        return [dict(item) for item in ds]
    
    def extract_answer(self, solution: str) -> str:
        """Extract numeric answer from GSM8K solution."""
        import re
        # GSM8K format: "#### 42"
        match = re.search(r'####\s*(\d+)', solution)
        if match:
            return match.group(1)
        return ""
    
    def process_item(self, raw: dict, idx: int) -> Optional[MathProblem]:
        problem_text = raw.get('question', '')
        solution_text = raw.get('answer', '')
        
        answer = self.extract_answer(solution_text)
        if not answer:
            return None
        
        return MathProblem(
            id=f"gsm8k_{idx}",
            problem=problem_text,
            answer=answer,
            solution=solution_text,
            source="gsm8k",
            difficulty=2,  # Grade school = easy
            subject="arithmetic",
            metadata={}
        )


# =============================================================================
# Countdown Processor (for comparison with current experiments)
# =============================================================================

class CountdownProcessor(DatasetProcessor):
    """Process Countdown dataset (what GRPO-Zero uses)."""
    
    @property
    def name(self) -> str:
        return "Countdown"
    
    def load_raw(self) -> List[dict]:
        import pandas as pd
        # Load from GRPO-Zero's data
        data_path = Path(__file__).parent.parent / "GRPO-Zero" / "Countdown-Tasks-3to4" / "data"
        if not data_path.exists():
            raise FileNotFoundError(f"Countdown data not found at {data_path}")
        
        df = pd.read_parquet(data_path)
        return df.to_dict('records')
    
    def process_item(self, raw: dict, idx: int) -> Optional[MathProblem]:
        nums = raw.get('nums', [])
        target = raw.get('target', 0)
        
        problem_text = f"Using the numbers {list(nums)}, create an equation that equals {target}."
        
        return MathProblem(
            id=f"countdown_{idx}",
            problem=problem_text,
            answer=str(target),
            solution=None,  # No solution provided
            source="countdown",
            difficulty=1,
            subject="arithmetic",
            metadata={'nums': list(nums), 'target': target}
        )


# =============================================================================
# Storage & Utilities
# =============================================================================

def save_dataset(problems: List[MathProblem], output_dir: str):
    """Save processed dataset to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL (one JSON per line)
    data_file = output_path / "data.jsonl"
    with open(data_file, 'w') as f:
        for p in problems:
            f.write(json.dumps(p.to_dict()) + '\n')
    
    # Save metadata
    meta = {
        'total_problems': len(problems),
        'sources': list(set(p.source for p in problems)),
        'difficulties': {str(k): sum(1 for p in problems if p.difficulty == k) 
                        for k in set(p.difficulty for p in problems)},
        'subjects': {k: sum(1 for p in problems if p.subject == k)
                    for k in set(p.subject for p in problems) if k},
    }
    with open(output_path / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Saved {len(problems)} problems to {output_path}")
    logger.info(f"Metadata: {meta}")


def load_dataset(input_dir: str) -> List[MathProblem]:
    """Load processed dataset from disk."""
    input_path = Path(input_dir)
    data_file = input_path / "data.jsonl"
    
    problems = []
    with open(data_file) as f:
        for line in f:
            problems.append(MathProblem.from_dict(json.loads(line)))
    
    return problems


def inspect_dataset(input_dir: str, n: int = 5):
    """Print sample problems for inspection."""
    problems = load_dataset(input_dir)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {input_dir}")
    print(f"Total: {len(problems)} problems")
    print(f"{'='*60}\n")
    
    # Show by difficulty if available
    by_diff = {}
    for p in problems:
        d = p.difficulty or 0
        if d not in by_diff:
            by_diff[d] = []
        by_diff[d].append(p)
    
    for diff in sorted(by_diff.keys()):
        print(f"--- Difficulty {diff} ({len(by_diff[diff])} problems) ---\n")
        for p in by_diff[diff][:n]:
            print(f"ID: {p.id}")
            print(f"Subject: {p.subject}")
            print(f"Problem: {p.problem[:200]}...")
            print(f"Answer: {p.answer}")
            print()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Math dataset preprocessing")
    parser.add_argument('--dataset', type=str, choices=['math', 'gsm8k', 'countdown'],
                       help='Dataset to process')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--levels', type=str, default=None,
                       help='MATH difficulty levels (e.g., "1,2,3")')
    parser.add_argument('--subjects', type=str, default=None,
                       help='MATH subjects (e.g., "Algebra,Geometry")')
    parser.add_argument('--max-items', type=int, default=None,
                       help='Max items to process')
    parser.add_argument('--inspect', type=str, default=None,
                       help='Inspect existing dataset')
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_dataset(args.inspect)
        return
    
    if not args.dataset:
        parser.print_help()
        return
    
    # Select processor
    if args.dataset == 'math':
        levels = [int(x) for x in args.levels.split(',')] if args.levels else None
        subjects = args.subjects.split(',') if args.subjects else None
        processor = MATHProcessor(levels=levels, subjects=subjects)
    elif args.dataset == 'gsm8k':
        processor = GSM8KProcessor()
    elif args.dataset == 'countdown':
        processor = CountdownProcessor()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Process
    problems = processor.process_all(max_items=args.max_items)
    
    # Save
    save_dataset(problems, args.output)
    
    # Quick sanity check
    print("\n--- Sample Output ---")
    for p in problems[:2]:
        print(f"Problem: {p.problem[:100]}...")
        print(f"Answer: {p.answer}")
        print()


if __name__ == "__main__":
    main()

