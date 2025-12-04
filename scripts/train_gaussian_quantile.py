"""
Gaussian Quantile Normalization for reward weighting.

Instead of just (R - mean) / std (matching first 2 moments),
we use quantile transformation to match ALL moments to Gaussian.

This maps rewards to their rank → quantile → Gaussian inverse CDF.
More stable optimization by enforcing full Gaussian shape.
"""
import os
import dataclasses
from collections import defaultdict
import numpy as np
from scipy import stats

import grpo

def normalize_rewards_gaussian_quantile(episodes):
    """
    Gaussian quantile normalization.
    
    Steps:
    1. Rank rewards within group
    2. Map ranks to quantiles (0, 1)
    3. Map quantiles to Gaussian inverse CDF (ppf)
    
    This ensures the output distribution is exactly Gaussian,
    matching ALL moments, not just mean/std.
    """
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    
    output = []
    for group in groups.values():
        n = len(group)
        group_rewards = np.array([item.reward for item in group])
        
        if n < 2:
            # Can't do quantile transform with 1 sample
            for episode in group:
                output.append(episode)
            continue
        
        # Get ranks (1 to n), handling ties with average
        ranks = stats.rankdata(group_rewards, method='average')
        
        # Map to quantiles in (0, 1) - avoid 0 and 1 for ppf stability
        # Use (rank - 0.5) / n which maps to (0.5/n, 1-0.5/n)
        quantiles = (ranks - 0.5) / n
        
        # Map to Gaussian via inverse CDF (ppf)
        # This gives us standard normal values
        gaussian_rewards = stats.norm.ppf(quantiles)
        
        for i, episode in enumerate(group):
            episode = dataclasses.replace(episode, reward=gaussian_rewards[i])
            output.append(episode)
    
    return output

# Monkey-patch
grpo.normalize_rewards_per_group = normalize_rewards_gaussian_quantile
print("[GAUSSIAN QUANTILE] Using quantile normalization (all moments → Gaussian)")

if __name__ == "__main__":
    from train import main
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)

