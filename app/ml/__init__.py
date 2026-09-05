"""Local ML components.

Keep this package __init__ free of torch: the self-evolution service imports
dataset_builder/taxonomy on the API's import path, where heavy dependencies
must stay optional. Torch-dependent members (repo_analyzer, model_registry
consumers) are imported lazily from their submodules.
"""
from app.ml.extractive import extractive_summary, tokenize
from app.ml.taxonomy import USE_CASE_TAXONOMY, match_use_cases

__all__ = [
    "extractive_summary",
    "tokenize",
    "USE_CASE_TAXONOMY",
    "match_use_cases",
]
