"""
Feature Engineering Consolidation Package
==========================================
Provides the unified, enterprise-grade `FeatureExtractor` class for transforming
raw integer token sequences into the full peak hybrid sparse-dense feature space
(Cumulative 1-5 TF-IDF + 34 Dense Descriptors, achieving 91.74% 5-Fold CV).

Also exports `Stage4FeatureExtractor` as an alias for backwards compatibility.
"""

from .feature_extractor import FeatureExtractor, Stage4FeatureExtractor
from .replicate_exp4_sota import resolve_dataset_paths, load_dataset

# Bind static convenience methods for dataset resolution and loading
FeatureExtractor.resolve_dataset_paths = staticmethod(resolve_dataset_paths)
FeatureExtractor.load_dataset = staticmethod(load_dataset)

__all__ = [
    "FeatureExtractor",
    "Stage4FeatureExtractor",
    "resolve_dataset_paths",
    "load_dataset",
]
