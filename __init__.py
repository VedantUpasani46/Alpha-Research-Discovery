"""
Alpha Research & Discovery Framework
=====================================

Production-quality tools for quantitative alpha research:

- walk_forward: Rolling walk-forward cross-validation engine
- alpha_combiner: Multiple alpha combination / meta-learning methods
- data_fetcher: Flexible data acquisition with caching and validation
"""

from framework.walk_forward import WalkForwardValidator, WalkForwardResult
from framework.alpha_combiner import (
    EqualWeightCombiner,
    ICWeightedCombiner,
    RidgeCombiner,
    LightGBMCombiner,
    OptimalShrinkageCombiner,
)
from framework.data_fetcher import DataFetcher

__all__ = [
    "WalkForwardValidator",
    "WalkForwardResult",
    "EqualWeightCombiner",
    "ICWeightedCombiner",
    "RidgeCombiner",
    "LightGBMCombiner",
    "OptimalShrinkageCombiner",
    "DataFetcher",
]
