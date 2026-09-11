"""Fixed OpenML benchmark suite definitions.

Every task is stratified-sampled to 500 rows when it is larger. That keeps each
five-fold TabPFN training context at 400 rows, making the CPU suite practical
while retaining shared, shape-compatible folds for GPU fused inference.
"""

from __future__ import annotations


SAMPLE_SIZE = 500
SAMPLING_SEED = 2026

DATASETS = {
    "credit_g": {
        "openml_data_id": 31,
        "name": "German Credit",
        "description": "Mixed-type credit-risk classification.",
        "positive_label": "good",
    },
    "adult_income": {
        "openml_data_id": 1590,
        "name": "Adult Income",
        "description": "Census attributes for predicting income above $50K.",
        "positive_label": ">50k",
    },
    "bank_marketing": {
        "openml_data_id": 1461,
        "name": "Bank Marketing",
        "description": "Direct-marketing attributes for predicting a term-deposit subscription.",
        "positive_label": "yes",
    },
    "spambase": {
        "openml_data_id": 44,
        "name": "Spambase",
        "description": "Email-word and character-frequency features for spam detection.",
        "positive_label": "1",
    },
    "phoneme": {
        "openml_data_id": 1489,
        "name": "Phoneme",
        "description": "Acoustic measurements for phoneme classification.",
        "positive_label": "1",
    },
}
