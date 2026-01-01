"""
Shared Pydantic models for API communication.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    """Task information."""
    task_id: str
    prompt_id: int
    model_name: str
    model_short_name: str
    status: TaskStatus
    dataset: str = "asap"  # "asap" or "toefl11"
    worker_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_completed_epoch: int = 0
    max_epochs: int = 30
    error_message: Optional[str] = None


class TaskStartRequest(BaseModel):
    """Request to start a task."""
    worker_id: str


class TaskStartResponse(BaseModel):
    """Response when starting a task."""
    success: bool
    task: Optional[TaskInfo] = None
    message: str = ""


class TaskCompleteRequest(BaseModel):
    """Request to complete a task."""
    worker_id: str
    summary: Dict[str, Any] = {}


class TaskFailRequest(BaseModel):
    """Request to mark a task as failed."""
    worker_id: str
    error_message: str


class EpochProgress(BaseModel):
    """Progress update for an epoch."""
    epoch: int
    training_loss: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    """Overall status response."""
    total_tasks: int
    pending: int
    running: int
    completed: int
    failed: int
    tasks: List[TaskInfo]


class CheckpointInfo(BaseModel):
    """Information about a checkpoint."""
    task_id: str
    epoch: int
    exists: bool
    size_bytes: Optional[int] = None


class ExperimentConfig(BaseModel):
    """Experiment configuration sent to workers.

    These are the default values. Override by setting values in the server's
    experiment_config.json file.
    """
    task_id: str
    prompt_id: int
    model_name: str
    dataset: str = "asap"  # "asap" or "toefl11"
    max_epochs: int = 30
    # Training hyperparameters (can be overridden by server config)
    lr: float = 1e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    weight_decay: float = 0.0  # L2 regularization
    max_seq_len: int = 2048
    batch_size: int = 1
    grad_accum_steps: int = 4
    seed: int = 42
    dev_M: int = 5
    dev_seed: int = 42
    # Training text format options
    include_output_format: bool = False  # Add scoring output format after essay
    # Training text mode:
    #   "essay_only" - Train on raw essay text only (original behavior)
    #   "scoring_prompt_short" - Train on shortened scoring prompt (task + score range + essay + output format)
    #   "scoring_prompt_full" - Train on full scoring prompt (with writing prompt + rubric)
    training_text_mode: str = "essay_only"
    # Memory optimization
    gradient_checkpointing: bool = False
    load_in_8bit: bool = False  # 8-bit quantization for memory savings
    # Experimental: Train on base model, score on instruct model
    # When True and model is llama8b, trains LoRA on Llama-3.1-8B (base)
    # then applies adapter to Llama-3.1-8B-Instruct for scoring
    train_on_base_model: bool = False
    # LR schedule options
    # "warmup_decay" - Linear warmup (5 epochs) + linear decay (25 epochs)
    # "exponential_warmup" - Exponential growth from lr_init to lr_final
    lr_schedule: str = "warmup_decay"
    lr_init: float = 1e-7  # Initial LR for exponential_warmup
    lr_final: float = 1e-5  # Final LR for exponential_warmup
