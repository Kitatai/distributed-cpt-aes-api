"""
Data loading module for TOEFL11 dataset.

Handles loading essays from the ETS Corpus of Non-Native Written English.
Provides the same interface as ASAPDataLoader for consistency.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .data_loader import EssayData, DatasetSplit

logger = logging.getLogger(__name__)

# Score level mapping
SCORE_LEVEL_MAP = {
    'low': 0,
    'medium': 1,
    'high': 2,
}


class TOEFL11DataLoader:
    """
    Data loader for TOEFL11 (ETS Corpus of Non-Native Written English) dataset.

    Supports:
    - Loading from index.csv + individual essay files
    - Filtering by prompt (P1-P8)
    - Creating dev/train splits with reproducible random sampling
    - Saving and loading split IDs for reproducibility
    """

    def __init__(
        self,
        data_dir: str,
        use_tokenized: bool = False,
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the ETS_Corpus_of_Non-Native_Written_English directory
            use_tokenized: Whether to use tokenized essays (default: False, use original)
        """
        self.data_dir = Path(data_dir)
        self.use_tokenized = use_tokenized

        # Set paths
        self.index_path = self.data_dir / "data" / "text" / "index.csv"
        self.responses_dir = self.data_dir / "data" / "text" / "responses"
        if use_tokenized:
            self.essays_dir = self.responses_dir / "tokenized"
        else:
            self.essays_dir = self.responses_dir / "original"

        self._df: Optional[pd.DataFrame] = None
        self._essays_cache: Dict[str, str] = {}

    def load(self) -> pd.DataFrame:
        """Load the index file."""
        if self._df is not None:
            return self._df

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        self._df = pd.read_csv(self.index_path)

        # Parse prompt ID from 'P1', 'P2', etc. to integer 1, 2, etc.
        self._df['prompt_id'] = self._df['Prompt'].str.extract(r'P(\d+)').astype(int)

        # Map score levels to integers
        self._df['score'] = self._df['Score Level'].map(SCORE_LEVEL_MAP)

        # Create essay_id from filename (remove .txt extension)
        self._df['essay_id'] = self._df['Filename'].str.replace('.txt', '', regex=False).astype(int)

        logger.info(f"Loaded {len(self._df)} essays from {self.index_path}")
        return self._df

    def _load_essay_text(self, filename: str) -> str:
        """Load essay text from file."""
        if filename in self._essays_cache:
            return self._essays_cache[filename]

        essay_path = self.essays_dir / filename
        if not essay_path.exists():
            logger.warning(f"Essay file not found: {essay_path}")
            return ""

        with open(essay_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        self._essays_cache[filename] = text
        return text

    def get_prompt_data(self, prompt_id: int) -> pd.DataFrame:
        """Get data for a specific prompt (1-8)."""
        df = self.load()
        prompt_df = df[df['prompt_id'] == prompt_id].copy()
        logger.info(f"TOEFL11 Prompt {prompt_id}: {len(prompt_df)} essays")
        return prompt_df

    def get_essays_for_prompt(self, prompt_id: int) -> List[EssayData]:
        """Get list of EssayData objects for a specific prompt."""
        df = self.get_prompt_data(prompt_id)
        essays = []
        for _, row in df.iterrows():
            essay_text = self._load_essay_text(row['Filename'])
            essay = EssayData(
                essay_id=int(row['essay_id']),
                essay_set=int(row['prompt_id']),
                essay_text=essay_text,
                score=int(row['score']),
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
            prompt_id: Prompt ID to filter by (1-8)
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
        scores = df['score'].dropna()
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
        prompt_ids = sorted(df['prompt_id'].unique())
        return {pid: self.get_prompt_info(pid) for pid in prompt_ids}


def load_toefl11_for_experiment(
    data_dir: str,
    prompt_id: int,
    dev_M: int = 30,
    seed: int = 42,
    splits_dir: Optional[str] = None,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit, str]:
    """
    Convenience function to load TOEFL11 data for an experiment.

    Args:
        data_dir: Path to ETS_Corpus_of_Non-Native_Written_English directory
        prompt_id: Prompt ID to filter by (1-8)
        dev_M: Number of essays for dev set
        seed: Random seed
        splits_dir: Directory to save/load splits

    Returns:
        Tuple of (dev_split, test_split, full_split, dev_ids_path)
    """
    loader = TOEFL11DataLoader(data_dir)

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
