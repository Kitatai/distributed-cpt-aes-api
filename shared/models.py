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
    """Experiment configuration sent to workers."""
    task_id: str
    prompt_id: int
    model_name: str
    dataset: str = "asap"  # "asap" or "toefl11"
    max_epochs: int = 30
    lr: float = 1e-6
    lora_r: int = 4
    lora_alpha: int = 16
    max_seq_len: int = 256
    batch_size: int = 1
    grad_accum_steps: int = 8
    seed: int = 42
    dev_M: int = 5
    dev_seed: int = 42
