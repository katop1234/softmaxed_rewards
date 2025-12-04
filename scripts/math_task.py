"""
MATH Dataset task for GRPO training.
Supports competition math problems (AMC/AIME level).
"""
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer

SYSTEM_MESSAGE = (
    "You are a helpful assistant that solves math problems. "
    "Think step by step and provide your final answer in \\boxed{}."
)

USER_TEMPLATE = "{problem}\n\nShow your reasoning in <think> </think> tags, then give your final answer in \\boxed{{}}."

RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


class MathDataset(Dataset):
    """MATH dataset for training."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        # Load from JSONL
        data_file = Path(data_path) / "data.jsonl"
        self.data = []
        with open(data_file) as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Split train/test
        if split == "train":
            self.data = self.data[:-test_size]
        else:
            self.data = self.data[-test_size:]
        
        self.tokenizer = tokenizer
        print(f"[MATH] Loaded {len(self.data)} {split} examples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.encode_prefix(item["problem"])
        return {
            "problem": item["problem"],
            "answer": item["answer"],
            "difficulty": item.get("difficulty"),
            "subject": item.get("subject"),
            **encoded
        }

    def encode_prefix(self, problem: str):
        """Encode the problem as model input."""
        user_message = USER_TEMPLATE.format(problem=problem)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        return MiniBatch(
            numbers=[item["problem"] for item in batch],  # Reuse 'numbers' field for problem
            target=[item["answer"] for item in batch],     # Reuse 'target' field for answer
            prefix=[item["prefix"] for item in batch],
            prefix_tokens=[item["prefix_tokens"] for item in batch],
            prefix_token_ids=[item["prefix_token_ids"] for item in batch],
        )


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...}."""
    # Handle nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    
    # Simpler pattern fallback
    simple = re.search(r'\\boxed\{([^}]+)\}', text)
    if simple:
        return simple.group(1).strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    
    # Remove whitespace
    answer = answer.strip()
    
    # Remove common LaTeX formatting
    answer = answer.replace("\\$", "")
    answer = answer.replace("$", "")
    answer = answer.replace("\\text{", "").replace("}", "")
    answer = answer.replace("\\mathrm{", "")
    answer = answer.replace("\\frac", "")
    answer = answer.replace("\\dfrac", "")
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace(" ", "")
    
    # Try to evaluate simple numeric expressions
    try:
        # Handle fractions like 1/2
        if "/" in answer and answer.replace("/", "").replace("-", "").replace(".", "").isdigit():
            parts = answer.split("/")
            if len(parts) == 2:
                answer = str(float(parts[0]) / float(parts[1]))
    except:
        pass
    
    return answer.lower()


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """Check if response follows format with <think> and \\boxed{}."""
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]
    
    reward = 0.0
    
    # Check for think tags
    if re.search(r"<think>.*?</think>", response, re.DOTALL):
        reward += 0.3
    
    # Check for boxed answer
    if re.search(r"\\boxed\{.+\}", response):
        reward += 0.7
    
    return reward


def answer_reward_function(response: str, ground_truth: str) -> float:
    """Check if the answer matches ground truth."""
    predicted = extract_boxed_answer(response)
    if predicted is None:
        return 0.0
    
    # Normalize both
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    # Exact match
    if pred_norm == gt_norm:
        return 1.0
    
    # Try numeric comparison
    try:
        pred_val = float(eval(pred_norm.replace("^", "**")))
        gt_val = float(eval(gt_norm.replace("^", "**")))
        if abs(pred_val - gt_val) < 1e-6:
            return 1.0
    except:
        pass
    
    return 0.0


def reward_function(
    response: str,
    numbers: str = None,  # This is actually the problem text
    target: str = None,   # This is the ground truth answer
    end_token: str = None,
) -> Dict[str, Any]:
    """Reward function for MATH dataset.
    
    Total reward = 0.1 * format_reward + answer_reward
    """
    format_reward = format_reward_function("<think>" + response, end_token)
    answer_reward = answer_reward_function(response, target)
    
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }

