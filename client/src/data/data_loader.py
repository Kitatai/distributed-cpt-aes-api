"""
Data loading module for ASAP dataset.

Handles loading, filtering by prompt, and splitting into dev/train sets.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EssayData:
    """Container for a single essay with its metadata."""
    essay_id: int
    essay_set: int
    essay_text: str
    score: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'essay_id': self.essay_id,
            'essay_set': self.essay_set,
            'essay_text': self.essay_text,
            'score': self.score,
        }


@dataclass
class DatasetSplit:
    """Container for dataset split."""
    essays: List[EssayData]
    essay_ids: List[int]

    def __len__(self) -> int:
        return len(self.essays)

    def get_texts(self) -> List[str]:
        return [e.essay_text for e in self.essays]

    def get_scores(self) -> List[Optional[int]]:
        return [e.score for e in self.essays]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self.essays])


class ASAPDataLoader:
    """
    Data loader for ASAP (Automated Student Assessment Prize) dataset.

    Supports:
    - Loading from TSV/CSV files
    - Filtering by prompt (essay_set)
    - Creating dev/train splits with reproducible random sampling
    - Saving and loading split IDs for reproducibility
    """

    def __init__(
        self,
        data_path: str,
        essay_column: str = "essay",
        score_column: str = "domain1_score",
        essay_set_column: str = "essay_set",
        essay_id_column: str = "essay_id",
    ):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the ASAP data file (TSV/CSV)
            essay_column: Column name for essay text
            score_column: Column name for score
            essay_set_column: Column name for prompt/essay set ID
            essay_id_column: Column name for essay ID
        """
        self.data_path = Path(data_path)
        self.essay_column = essay_column
        self.score_column = score_column
        self.essay_set_column = essay_set_column
        self.essay_id_column = essay_id_column

        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load the dataset from file."""
        if self._df is not None:
            return self._df

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Detect file format
        suffix = self.data_path.suffix.lower()
        if suffix == '.tsv':
            self._df = pd.read_csv(self.data_path, sep='\t', encoding='latin-1')
        elif suffix == '.csv':
            self._df = pd.read_csv(self.data_path, encoding='latin-1')
        else:
            # Try TSV first, then CSV
            try:
                self._df = pd.read_csv(self.data_path, sep='\t', encoding='latin-1')
            except Exception:
                self._df = pd.read_csv(self.data_path, encoding='latin-1')

        logger.info(f"Loaded {len(self._df)} essays from {self.data_path}")
        return self._df

    def get_prompt_data(self, prompt_id: int) -> pd.DataFrame:
        """Get data for a specific prompt (essay_set)."""
        df = self.load()
        prompt_df = df[df[self.essay_set_column] == prompt_id].copy()
        logger.info(f"Prompt {prompt_id}: {len(prompt_df)} essays")
        return prompt_df

    def get_essays_for_prompt(self, prompt_id: int) -> List[EssayData]:
        """Get list of EssayData objects for a specific prompt."""
        df = self.get_prompt_data(prompt_id)
        essays = []
        for _, row in df.iterrows():
            essay = EssayData(
                essay_id=int(row[self.essay_id_column]),
                essay_set=int(row[self.essay_set_column]),
                essay_text=str(row[self.essay_column]),
                score=int(row[self.score_column]) if pd.notna(row[self.score_column]) else None,
            )
            essays.append(essay)
        return essays

    def create_dev_split(
        self,
        prompt_id: int,
        M: int = 30,
        seed: int = 42,
        save_path: Optional[str] = None,
    ) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """
        Create dev, test, and full dataset splits.

        The dev split contains M randomly sampled essays (for epoch selection).
        The test split contains all essays EXCEPT dev (for final evaluation).
        The full split contains ALL essays (for continual pre-training).

        Args:
            prompt_id: Prompt ID to filter by
            M: Number of essays for dev set
            seed: Random seed for reproducibility
            save_path: Path to save dev IDs for reproducibility

        Returns:
            Tuple of (dev_split, test_split, full_split)
        """
        essays = self.get_essays_for_prompt(prompt_id)
        n_essays = len(essays)

        if M > n_essays:
            logger.warning(f"M={M} > n_essays={n_essays}, using all essays for dev")
            M = n_essays

        # Random sampling for dev set
        rng = np.random.RandomState(seed)
        all_indices = np.arange(n_essays)
        dev_indices = set(rng.choice(all_indices, size=M, replace=False))
        test_indices = [i for i in all_indices if i not in dev_indices]

        # Create splits
        dev_essays = [essays[i] for i in sorted(dev_indices)]
        test_essays = [essays[i] for i in test_indices]
        dev_ids = [e.essay_id for e in dev_essays]
        test_ids = [e.essay_id for e in test_essays]

        dev_split = DatasetSplit(essays=dev_essays, essay_ids=dev_ids)
        test_split = DatasetSplit(essays=test_essays, essay_ids=test_ids)
        full_split = DatasetSplit(essays=essays, essay_ids=[e.essay_id for e in essays])

        # Save dev IDs if path provided
        if save_path:
            self.save_dev_ids(dev_ids, save_path)

        logger.info(f"Created splits: dev={len(dev_split)}, test={len(test_split)}, full={len(full_split)}")
        return dev_split, test_split, full_split

    def load_existing_split(
        self,
        prompt_id: int,
        dev_ids_path: str,
    ) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """
        Load existing dev split from saved IDs.

        Args:
            prompt_id: Prompt ID to filter by
            dev_ids_path: Path to JSON file with dev IDs

        Returns:
            Tuple of (dev_split, test_split, full_split)
        """
        essays = self.get_essays_for_prompt(prompt_id)
        essay_dict = {e.essay_id: e for e in essays}

        # Load dev IDs
        with open(dev_ids_path, 'r') as f:
            dev_ids = json.load(f)

        dev_ids_set = set(dev_ids)

        # Create splits
        dev_essays = [essay_dict[eid] for eid in dev_ids if eid in essay_dict]
        test_essays = [e for e in essays if e.essay_id not in dev_ids_set]

        dev_split = DatasetSplit(essays=dev_essays, essay_ids=[e.essay_id for e in dev_essays])
        test_split = DatasetSplit(essays=test_essays, essay_ids=[e.essay_id for e in test_essays])
        full_split = DatasetSplit(essays=essays, essay_ids=[e.essay_id for e in essays])

        logger.info(f"Loaded splits: dev={len(dev_split)}, test={len(test_split)}, full={len(full_split)}")
        return dev_split, test_split, full_split

    @staticmethod
    def save_dev_ids(dev_ids: List[int], path: str):
        """Save dev IDs to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dev_ids, f, indent=2)
        logger.info(f"Saved dev IDs to {path}")

    @staticmethod
    def load_dev_ids(path: str) -> List[int]:
        """Load dev IDs from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def get_prompt_info(self, prompt_id: int) -> Dict:
        """Get information about a specific prompt."""
        df = self.get_prompt_data(prompt_id)
        scores = df[self.score_column].dropna()
        return {
            'prompt_id': prompt_id,
            'n_essays': len(df),
            'score_min': int(scores.min()),
            'score_max': int(scores.max()),
            'score_mean': float(scores.mean()),
            'score_std': float(scores.std()),
        }

    def get_all_prompts_info(self) -> Dict[int, Dict]:
        """Get information about all prompts."""
        df = self.load()
        prompt_ids = sorted(df[self.essay_set_column].unique())
        return {pid: self.get_prompt_info(pid) for pid in prompt_ids}


class ContinualPretrainingDataset:
    """
    Dataset wrapper for continual pre-training.

    Provides essay texts without labels for language modeling.
    """

    def __init__(self, essays: List[EssayData]):
        """
        Initialize the dataset.

        Args:
            essays: List of EssayData objects
        """
        self.essays = essays
        self.texts = [e.essay_text for e in essays]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]

    def get_all_texts(self) -> List[str]:
        """Get all essay texts."""
        return self.texts


def load_asap_for_experiment(
    data_path: str,
    prompt_id: int,
    dev_M: int = 30,
    seed: int = 42,
    splits_dir: Optional[str] = None,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit, str]:
    """
    Convenience function to load ASAP data for an experiment.

    Args:
        data_path: Path to ASAP data file
        prompt_id: Prompt ID to filter by
        dev_M: Number of essays for dev set
        seed: Random seed
        splits_dir: Directory to save/load splits

    Returns:
        Tuple of (dev_split, test_split, full_split, dev_ids_path)
    """
    loader = ASAPDataLoader(data_path)

    if splits_dir:
        splits_path = Path(splits_dir) / f"prompt_{prompt_id}"
        splits_path.mkdir(parents=True, exist_ok=True)
        dev_ids_path = splits_path / "dev_ids.json"

        if dev_ids_path.exists():
            logger.info(f"Loading existing split from {dev_ids_path}")
            dev_split, test_split, full_split = loader.load_existing_split(prompt_id, str(dev_ids_path))
        else:
            logger.info(f"Creating new split, saving to {dev_ids_path}")
            dev_split, test_split, full_split = loader.create_dev_split(
                prompt_id, M=dev_M, seed=seed, save_path=str(dev_ids_path)
            )
    else:
        dev_split, test_split, full_split = loader.create_dev_split(
            prompt_id, M=dev_M, seed=seed
        )
        dev_ids_path = None

    return dev_split, test_split, full_split, str(dev_ids_path) if dev_ids_path else None
