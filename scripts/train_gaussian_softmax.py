"""
Gaussian Quantile + Softmax weighting.

First apply Gaussian quantile transform (all moments → Gaussian),
then apply softmax weighting on the Gaussianized rewards.

This combines:
1. Stability from Gaussian quantile (full distribution matching)
2. Focus on best samples from softmax weighting
"""
import os
import dataclasses
from collections import defaultdict
import numpy as np
from scipy import stats

import grpo

SOFTMAX_TAU = float(os.environ.get("SOFTMAX_TAU", "0.5"))

def normalize_rewards_gaussian_softmax(episodes, temperature=None):
    """
    Gaussian quantile normalization followed by softmax weighting.
    """
    tau = temperature if temperature else SOFTMAX_TAU
    
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    
    output = []
    for group in groups.values():
        n = len(group)
        group_rewards = np.array([item.reward for item in group])
        
        if n < 2:
            for episode in group:
                output.append(episode)
            continue
        
        # Step 1: Gaussian quantile transform
        ranks = stats.rankdata(group_rewards, method='average')
        quantiles = (ranks - 0.5) / n
        gaussian_rewards = stats.norm.ppf(quantiles)
        
        # Step 2: Softmax weighting on Gaussianized rewards
        # Now rewards are in standard normal range (~-2 to +2)
        # τ=1.0 should work well here since rewards are normalized
        scaled = gaussian_rewards / tau
        scaled = scaled - np.max(scaled)  # numerical stability
        exp_r = np.exp(scaled)
        weights = exp_r / (np.sum(exp_r) + 1e-8)
        normalized = weights * n
        
        for i, episode in enumerate(group):
            episode = dataclasses.replace(episode, reward=normalized[i])
            output.append(episode)
    
    return output

# Monkey-patch
grpo.normalize_rewards_per_group = normalize_rewards_gaussian_softmax
print(f"[GAUSSIAN+SOFTMAX] Quantile → Softmax with τ = {SOFTMAX_TAU}")

if __name__ == "__main__":
    from train import main
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)

