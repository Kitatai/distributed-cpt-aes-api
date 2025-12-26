"""
Simple training loop without HuggingFace Trainer.

Uses a single model instance throughout training and evaluation.
"""

import torch
import logging
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import json
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


@dataclass
class SimpleTrainerConfig:
    """Configuration for simple trainer."""
    lr: float = 1e-6
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = None
    max_seq_len: int = 256
    batch_size: int = 1
    grad_accum_steps: int = 8
    seed: int = 42  # Random seed for reproducibility

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class EssayLMDataset(Dataset):
    """Simple dataset for language modeling on essays."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.encodings = []
        logger.info(f"Tokenizing {len(texts)} texts...")

        for text in tqdm(texts, desc="Tokenizing"):
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt",
            )
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

        logger.info(f"Tokenization complete: {len(self.encodings)} examples")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def collate_fn(batch, pad_token_id: int):
    """Collate function with dynamic padding."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Pad on the right
        ids = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        mask = torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ])

        # Labels: -100 for padding (ignored in loss)
        lbl = ids.clone()
        lbl[mask == 0] = -100

        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(lbl)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


class SimpleLoRATrainer:
    """
    Simple LoRA trainer that keeps model in memory throughout.

    Uses basic PyTorch training loop instead of HuggingFace Trainer.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: SimpleTrainerConfig,
        output_dir: str,
    ):
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.optimizer = None
        self._lora_applied = False

    def apply_lora(self):
        """Apply LoRA adapter to the model."""
        if self._lora_applied:
            logger.info("LoRA already applied")
            return

        logger.info("Applying LoRA adapter...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )

        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
        )

        self._lora_applied = True
        logger.info("LoRA adapter applied")

    def get_model(self):
        """Get the model (with LoRA if applied)."""
        return self.model if self._lora_applied else self.base_model

    def train_epoch(self, texts: List[str], epoch: int) -> Dict:
        """
        Train for one epoch.

        Args:
            texts: List of essay texts for training
            epoch: Current epoch number (1-indexed)

        Returns:
            Dictionary with training stats
        """
        if not self._lora_applied:
            self.apply_lora()

        self.model.train()

        # Create dataset and dataloader
        dataset = EssayLMDataset(
            texts,
            self.tokenizer,
            max_length=self.config.max_seq_len
        )

        # Use generator for reproducible shuffling
        # Seed varies by epoch for different order each epoch, but reproducible
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + epoch)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.tokenizer.pad_token_id),
            generator=generator,
        )

        total_loss = 0.0
        num_steps = 0
        self.optimizer.zero_grad()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            labels = batch["labels"].to(self.model.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / self.config.grad_accum_steps
            loss.backward()

            total_loss += outputs.loss.item()
            num_steps += 1

            # Gradient accumulation
            if (step + 1) % self.config.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        # Final optimizer step if needed
        if num_steps % self.config.grad_accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
        logger.info(f"Epoch {epoch} completed: avg_loss={avg_loss:.4f}")

        # Save adapter
        adapter_path = self.output_dir / f"epoch_{epoch}" / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(adapter_path))
        logger.info(f"Saved adapter to {adapter_path}")

        # Save training info
        info = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "num_steps": num_steps,
            "config": {
                "lr": self.config.lr,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "max_seq_len": self.config.max_seq_len,
                "batch_size": self.config.batch_size,
                "grad_accum_steps": self.config.grad_accum_steps,
            },
        }
        with open(adapter_path.parent / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        return {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "num_steps": num_steps,
            "adapter_path": str(adapter_path),
        }

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        model = self.get_model()
        model.eval()

    def set_train_mode(self):
        """Set model to training mode."""
        if self._lora_applied:
            self.model.train()
