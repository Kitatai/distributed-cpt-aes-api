"""
Configuration module for Zero-shot Essay Scoring with Continual Pre-training.

This module defines all configuration classes and constants for the experiment.
Designed for research use with easy parameter modification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import json


# ASAP Prompt-specific score ranges (domain1_score)
ASAP_SCORE_RANGES: Dict[int, Tuple[int, int]] = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
}

# Supported models
SUPPORTED_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen3-8B",
]


@dataclass
class DataConfig:
    """Configuration for dataset handling."""
    dataset: str = "asap"
    prompt_id: int = 1
    data_path: str = "data/asap/training_set_rel3.tsv"
    essay_column: str = "essay"
    score_column: str = "domain1_score"
    essay_set_column: str = "essay_set"
    essay_id_column: str = "essay_id"

    @property
    def score_range(self) -> Tuple[int, int]:
        """Get score range for current prompt."""
        return ASAP_SCORE_RANGES.get(self.prompt_id, (0, 10))

    @property
    def y_min(self) -> int:
        return self.score_range[0]

    @property
    def y_max(self) -> int:
        return self.score_range[1]


@dataclass
class PromptingConfig:
    """Configuration for scoring prompt construction."""
    rubric_type: str = "generic_holistic_v1"
    rubric_text: str = ""  # Empty for now, can be filled later per-prompt
    include_reasoning: bool = False
    output_prefix: str = "The score of this essay: "
    output_requires_newline: bool = True
    newline_delimiter: str = "\n"
    assistant_prefill: bool = True

    # Prompt-specific rubrics (to be filled later)
    prompt_rubrics: Dict[int, str] = field(default_factory=dict)


@dataclass
class DecodingConfig:
    """Configuration for greedy decoding (deterministic score extraction)."""
    do_sample: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 12
    stop_on_newline: bool = True


@dataclass
class LogitExtractionConfig:
    """Configuration for logit-based expected value extraction.

    The UnifiedLogitExtractor uses probability-based pruning:
    - Explores token paths (space, digits, newline)
    - Prunes paths when cumulative probability < prob_threshold
    - Accumulates probabilities for all paths leading to same score
    """
    enabled: bool = False  # Disabled for faster greedy-only scoring
    max_steps: int = 4  # Maximum search depth for token paths
    prob_threshold: float = 1e-6  # Minimum probability to continue exploration


@dataclass
class DevSplitConfig:
    """Configuration for development data split."""
    M: int = 30  # Number of samples for dev set
    seed: int = 42
    include_dev_in_cpt: bool = True  # Include dev essays in continual pre-training


@dataclass
class ContinualPretrainingConfig:
    """Configuration for label-free continual pre-training with LoRA."""
    method: str = "lora"
    lr: float = 1e-6
    max_epochs: int = 30
    lora_r: int = 4
    lora_alpha: int = 16
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    max_seq_len: int = 2048
    batch_size: int = 1
    grad_accum_steps: int = 1
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    save_every_epoch: bool = True
    gradient_checkpointing: bool = False  # Enable to reduce memory usage


@dataclass
class SelectionConfig:
    """Configuration for epoch selection."""
    metric_for_e_star: str = "spearman"  # "spearman", "pearson", "qwk"
    selection_target: str = "y_hat_greedy"  # "y_hat_greedy" or "y_tilde_round"


@dataclass
class OutputConfig:
    """Configuration for output paths and formats."""
    base_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    splits_dir: str = "splits"

    # Output file patterns
    predictions_pattern: str = "predictions_epoch_{epoch}.csv"
    metrics_pattern: str = "metrics_epoch_{epoch}.json"
    score_dist_pattern: str = "score_dist_epoch_{epoch}.jsonl"

    def get_output_dir(self, model_name: str, dataset: str, prompt_id: int) -> Path:
        """Get output directory for specific experiment."""
        model_short = model_name.split("/")[-1]
        return Path(self.base_dir) / model_short / dataset / f"prompt_{prompt_id}"

    def get_checkpoint_dir(self, model_name: str, dataset: str, prompt_id: int, epoch: int) -> Path:
        """Get checkpoint directory for specific epoch."""
        model_short = model_name.split("/")[-1]
        return Path(self.checkpoint_dir) / model_short / dataset / f"prompt_{prompt_id}" / f"epoch_{epoch}" / "adapter"

    def get_splits_dir(self, dataset: str, prompt_id: int) -> Path:
        """Get splits directory for specific prompt."""
        return Path(self.splits_dir) / dataset / f"prompt_{prompt_id}"


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all sub-configs."""
    # Experiment identification
    experiment_name: str = "zeroshot_cpt_aes"
    seed: int = 42

    # Model
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    prompting: PromptingConfig = field(default_factory=PromptingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    logit_extraction: LogitExtractionConfig = field(default_factory=LogitExtractionConfig)
    dev_split: DevSplitConfig = field(default_factory=DevSplitConfig)
    cpt: ContinualPretrainingConfig = field(default_factory=ContinualPretrainingConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Device settings
    device: str = "cuda"
    dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        # Extract sub-configs
        data_cfg = DataConfig(**config_dict.pop('data', {}))
        prompting_cfg = PromptingConfig(**config_dict.pop('prompting', {}))
        decoding_cfg = DecodingConfig(**config_dict.pop('zeroshot_decoding', config_dict.pop('decoding', {})))
        logit_cfg = LogitExtractionConfig(**config_dict.pop('logit_expected_extraction', config_dict.pop('logit_extraction', {})))
        dev_cfg = DevSplitConfig(**config_dict.pop('dev_split', {}))
        cpt_cfg = ContinualPretrainingConfig(**config_dict.pop('continual_pretraining', config_dict.pop('cpt', {})))
        selection_cfg = SelectionConfig(**config_dict.pop('selection', {}))
        output_cfg = OutputConfig(**config_dict.pop('output', {}))

        return cls(
            data=data_cfg,
            prompting=prompting_cfg,
            decoding=decoding_cfg,
            logit_extraction=logit_cfg,
            dev_split=dev_cfg,
            cpt=cpt_cfg,
            selection=selection_cfg,
            output=output_cfg,
            **config_dict
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'model_name': self.model_name,
            'device': self.device,
            'dtype': self.dtype,
            'data': {
                'dataset': self.data.dataset,
                'prompt_id': self.data.prompt_id,
                'data_path': self.data.data_path,
                'score_range': list(self.data.score_range),
            },
            'prompting': {
                'rubric_type': self.prompting.rubric_type,
                'rubric_text': self.prompting.rubric_text,
                'include_reasoning': self.prompting.include_reasoning,
                'output_prefix': self.prompting.output_prefix,
                'assistant_prefill': self.prompting.assistant_prefill,
            },
            'decoding': {
                'do_sample': self.decoding.do_sample,
                'temperature': self.decoding.temperature,
                'max_new_tokens': self.decoding.max_new_tokens,
            },
            'logit_extraction': {
                'enabled': self.logit_extraction.enabled,
                'max_steps': self.logit_extraction.max_steps,
                'prob_threshold': self.logit_extraction.prob_threshold,
            },
            'dev_split': {
                'M': self.dev_split.M,
                'seed': self.dev_split.seed,
            },
            'cpt': {
                'method': self.cpt.method,
                'lr': self.cpt.lr,
                'max_epochs': self.cpt.max_epochs,
                'lora_r': self.cpt.lora_r,
                'lora_alpha': self.cpt.lora_alpha,
                'target_modules': self.cpt.target_modules,
            },
            'selection': {
                'metric_for_e_star': self.selection.metric_for_e_star,
                'selection_target': self.selection.selection_target,
            },
        }

    def save_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def save_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def create_default_config(prompt_id: int = 1, model_name: str = None) -> ExperimentConfig:
    """Create default configuration for a specific prompt."""
    config = ExperimentConfig()
    config.data.prompt_id = prompt_id
    if model_name:
        config.model_name = model_name
    return config


def create_config_for_all_prompts(model_name: str = None) -> Dict[int, ExperimentConfig]:
    """Create configurations for all ASAP prompts."""
    configs = {}
    for prompt_id in ASAP_SCORE_RANGES.keys():
        configs[prompt_id] = create_default_config(prompt_id, model_name)
    return configs
