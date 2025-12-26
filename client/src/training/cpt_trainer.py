"""
Continual Pre-training module with LoRA.

Implements label-free continual pre-training on essay texts
using next-token prediction (language modeling) objective.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from datasets import Dataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class CPTConfig:
    """Configuration for continual pre-training."""
    lr: float = 1e-6
    max_epochs: int = 30
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = None
    max_seq_len: int = 2048
    batch_size: int = 4
    grad_accum_steps: int = 8
    weight_decay: float = 0.01
    save_every_epoch: bool = True
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False
    seed: int = 42  # Random seed for reproducibility

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class EssayDataset(torch.utils.data.Dataset):
    """Dataset for essay texts."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


class ContinualPretrainer:
    """
    Continual pre-training with LoRA for essay domain adaptation.

    Uses label-free language modeling objective on essay texts.
    """

    def __init__(
        self,
        model_name: str,
        config: CPTConfig,
        output_dir: str,
        device: str = "cuda",
    ):
        """
        Initialize the trainer.

        Args:
            model_name: HuggingFace model name or path
            config: CPT configuration
            output_dir: Directory to save checkpoints
            device: Device to use
        """
        self.model_name = model_name
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[Trainer] = None
        self._is_loaded = False
        self._last_adapter_path: Optional[str] = None

    def load_model(self, adapter_path: Optional[str] = None):
        """
        Load base model and apply LoRA.

        Args:
            adapter_path: Optional path to existing adapter to continue training from
        """
        logger.info(f"Loading model: {self.model_name}")
        if adapter_path:
            logger.info(f"Will continue from adapter: {adapter_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine dtype
        dtype = torch.bfloat16 if self.config.bf16 else (
            torch.float16 if self.config.fp16 else torch.float32
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        if adapter_path and Path(adapter_path).exists():
            # Load existing adapter to continue training
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=True,
            )
            logger.info(f"Loaded existing adapter from {adapter_path}")
        else:
            # Apply new LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        self._is_loaded = True
        self._last_adapter_path = adapter_path
        logger.info("Model loaded with LoRA adapter")

    def prepare_dataset(self, texts: List[str]) -> Dataset:
        """
        Prepare dataset for training.

        Args:
            texts: List of essay texts

        Returns:
            HuggingFace Dataset
        """
        import gc

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_len,
                padding=False,
            )

        dataset = Dataset.from_dict({"text": texts})
        # Use small batch size and no parallelization to reduce memory
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,  # Process in smaller batches
            num_proc=1,  # Single process to avoid memory overhead
            remove_columns=["text"],
        )
        gc.collect()
        return tokenized

    def train(
        self,
        texts: List[str],
        resume_from_epoch: int = 0,
    ) -> Dict[int, str]:
        """
        Run continual pre-training.

        Args:
            texts: List of essay texts
            resume_from_epoch: Resume training from this epoch

        Returns:
            Dictionary mapping epoch to checkpoint path
        """
        if not self._is_loaded:
            self.load_model()

        logger.info(f"Starting CPT with {len(texts)} essays")

        # Prepare dataset
        dataset = self.prepare_dataset(texts)

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint paths
        checkpoint_paths: Dict[int, str] = {}

        # Training arguments for single epoch
        effective_batch_size = self.config.batch_size * self.config.grad_accum_steps
        steps_per_epoch = max(1, len(dataset) // effective_batch_size)

        for epoch in range(resume_from_epoch, self.config.max_epochs):
            logger.info(f"Training epoch {epoch + 1}/{self.config.max_epochs}")

            epoch_output_dir = self.output_dir / f"epoch_{epoch + 1}"

            training_args = TrainingArguments(
                output_dir=str(epoch_output_dir),
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.grad_accum_steps,
                learning_rate=self.config.lr,
                lr_scheduler_type="constant",  # No decay within epoch
                weight_decay=self.config.weight_decay,
                warmup_ratio=0.0,  # No warmup (constant LR)
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                gradient_checkpointing=self.config.gradient_checkpointing,
                logging_steps=max(1, steps_per_epoch // 10),
                save_strategy="no",  # We save manually
                report_to="none",
                dataloader_pin_memory=True,
                remove_unused_columns=False,
                seed=self.config.seed,  # Reproducibility
                data_seed=self.config.seed,  # Shuffle order reproducibility
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )

            # Train for one epoch
            train_result = trainer.train()
            logger.info(f"Epoch {epoch + 1} loss: {train_result.training_loss:.4f}")

            # Save adapter
            if self.config.save_every_epoch:
                adapter_path = epoch_output_dir / "adapter"
                adapter_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(adapter_path))
                checkpoint_paths[epoch + 1] = str(adapter_path)
                logger.info(f"Saved adapter to {adapter_path}")

                # Save training info
                info = {
                    "epoch": epoch + 1,
                    "loss": train_result.training_loss,
                    "model_name": self.model_name,
                    "config": {
                        "lr": self.config.lr,
                        "lora_r": self.config.lora_r,
                        "lora_alpha": self.config.lora_alpha,
                        "max_seq_len": self.config.max_seq_len,
                    },
                    "n_essays": len(texts),
                }
                with open(epoch_output_dir / "training_info.json", "w") as f:
                    json.dump(info, f, indent=2)

        return checkpoint_paths

    def train_single_epoch(
        self,
        texts: List[str],
        epoch: int,
    ) -> str:
        """
        Train for a single epoch.

        Args:
            texts: List of essay texts
            epoch: Current epoch number (1-indexed)

        Returns:
            Path to saved adapter
        """
        if not self._is_loaded:
            self.load_model()

        logger.info(f"Preparing dataset with {len(texts)} texts...")
        import sys
        sys.stdout.flush()
        sys.stderr.flush()

        dataset = self.prepare_dataset(texts)
        logger.info(f"Dataset prepared: {len(dataset)} examples")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        epoch_output_dir = self.output_dir / f"epoch_{epoch}"
        epoch_output_dir.mkdir(parents=True, exist_ok=True)

        effective_batch_size = self.config.batch_size * self.config.grad_accum_steps
        steps_per_epoch = max(1, len(dataset) // effective_batch_size)

        training_args = TrainingArguments(
            output_dir=str(epoch_output_dir),
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.grad_accum_steps,
            learning_rate=self.config.lr,
            lr_scheduler_type="constant",  # No decay within epoch
            weight_decay=self.config.weight_decay,
            warmup_ratio=0.0,  # No warmup (constant LR)
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            logging_steps=max(1, steps_per_epoch // 10),
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            seed=self.config.seed,  # Reproducibility
            data_seed=self.config.seed,  # Shuffle order reproducibility
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        train_result = self.trainer.train()
        logger.info(f"Epoch {epoch} loss: {train_result.training_loss:.4f}")

        # Save adapter
        adapter_path = epoch_output_dir / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(adapter_path))

        return str(adapter_path)

    def get_model(self) -> PreTrainedModel:
        """Get the current model."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer


def run_continual_pretraining(
    model_name: str,
    texts: List[str],
    output_dir: str,
    lr: float = 1e-6,
    max_epochs: int = 30,
    lora_r: int = 4,
    lora_alpha: int = 16,
    max_seq_len: int = 2048,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
) -> Dict[int, str]:
    """
    Convenience function to run continual pre-training.

    Args:
        model_name: HuggingFace model name
        texts: List of essay texts
        output_dir: Output directory
        lr: Learning rate
        max_epochs: Maximum epochs
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        grad_accum_steps: Gradient accumulation steps

    Returns:
        Dictionary mapping epoch to checkpoint path
    """
    config = CPTConfig(
        lr=lr,
        max_epochs=max_epochs,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
    )

    trainer = ContinualPretrainer(
        model_name=model_name,
        config=config,
        output_dir=output_dir,
    )

    return trainer.train(texts)
