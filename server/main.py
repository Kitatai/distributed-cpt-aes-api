"""
FastAPI server for distributed CPT-AES experiments.

Provides endpoints for:
- Task management (get next task, start, complete, fail)
- Checkpoint storage (upload/download adapters)
- Results storage (upload/download metrics and predictions)
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import aiofiles
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from filelock import FileLock
from pydantic import BaseModel

# Add shared models path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from models import (
    TaskStatus, TaskInfo, TaskStartRequest, TaskStartResponse,
    TaskCompleteRequest, TaskFailRequest, StatusResponse,
    CheckpointInfo, ExperimentConfig,
)

# Configuration
SERVER_DIR = Path(__file__).parent
DATA_DIR = SERVER_DIR / "data"
TASKS_DIR = DATA_DIR / "tasks"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
RESULTS_DIR = DATA_DIR / "results"
ASAP_DATA_DIR = DATA_DIR / "asap"
TOEFL11_DATA_DIR = DATA_DIR / "TOEFL11" / "ETS_Corpus_of_Non-Native_Written_English"

# Create directories
for d in [DATA_DIR, TASKS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, ASAP_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Models configuration
MODELS = [
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "llama8b"),
    ("meta-llama/Llama-3.2-3B-Instruct", "llama3b"),
    ("Qwen/Qwen3-8B", "qwen"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
]

# ASAP score ranges
ASAP_SCORE_RANGES = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60),
}

# TOEFL11 score ranges (low=0, medium=1, high=2 for all prompts)
TOEFL11_SCORE_RANGES = {
    1: (0, 2),
    2: (0, 2),
    3: (0, 2),
    4: (0, 2),
    5: (0, 2),
    6: (0, 2),
    7: (0, 2),
    8: (0, 2),
}


def get_task_file(task_id: str) -> Path:
    """Get path to task file."""
    return TASKS_DIR / f"{task_id}.json"


def load_task(task_id: str) -> Optional[TaskInfo]:
    """Load task from file."""
    task_file = get_task_file(task_id)
    if not task_file.exists():
        return None
    with open(task_file, 'r') as f:
        data = json.load(f)
    return TaskInfo(**data)


def save_task(task: TaskInfo):
    """Save task to file."""
    task_file = get_task_file(task.task_id)
    with open(task_file, 'w') as f:
        json.dump(task.model_dump(mode='json'), f, indent=2, default=str)


def get_all_tasks() -> list[TaskInfo]:
    """Get all tasks."""
    tasks = []
    for task_file in TASKS_DIR.glob("*.json"):
        with open(task_file, 'r') as f:
            data = json.load(f)
        tasks.append(TaskInfo(**data))
    return sorted(tasks, key=lambda t: t.task_id)


def get_verified_last_completed_epoch(task_id: str) -> int:
    """
    Get the last epoch where BOTH checkpoint AND metrics exist.
    This ensures data integrity - an epoch is only 'completed' if all data is saved.
    Returns -1 if no complete epochs exist.
    """
    checkpoint_dir = CHECKPOINTS_DIR / task_id
    results_dir = RESULTS_DIR / task_id

    # Find epochs with checkpoints (stored as zip files or directories)
    checkpoint_epochs = set()
    if checkpoint_dir.exists():
        for item in checkpoint_dir.iterdir():
            if item.name.startswith("epoch_"):
                try:
                    epoch = int(item.name.split("_")[1])
                    # Check for zip file or adapter directory
                    if item.is_dir() and (item / "adapter.zip").exists():
                        checkpoint_epochs.add(epoch)
                    elif item.is_dir() and (item / "adapter").is_dir():
                        checkpoint_epochs.add(epoch)
                except (ValueError, IndexError):
                    pass

    # Find epochs with metrics
    metrics_epochs = set()
    if results_dir.exists():
        for f in results_dir.glob("metrics_epoch_*.json"):
            try:
                epoch = int(f.stem.split("_")[-1])
                metrics_epochs.add(epoch)
            except (ValueError, IndexError):
                pass

    # Only count epochs where BOTH exist (epoch 0 has no checkpoint, only metrics)
    complete_epochs = set()
    for epoch in metrics_epochs:
        if epoch == 0 or epoch in checkpoint_epochs:
            complete_epochs.add(epoch)

    if not complete_epochs:
        return -1

    # Return the max epoch, but ensure all previous epochs are also complete
    max_epoch = max(complete_epochs)
    for e in range(1, max_epoch + 1):
        if e not in complete_epochs:
            # Gap found - return the last complete epoch before the gap
            return e - 1

    return max_epoch


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Server starting...")
    logger.info(f"Data directory: {DATA_DIR}")
    yield
    logger.info("Server shutting down...")


app = FastAPI(
    title="CPT-AES Experiment Server",
    description="API server for distributed continual pre-training experiments",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================
# Task Management Endpoints
# ============================================

@app.post("/tasks/init")
async def init_tasks(
    force: bool = Query(False, description="Force reinitialize all tasks"),
    dataset: str = Query("asap", description="Dataset to initialize (asap or toefl11)")
):
    """Initialize experiment tasks for a specific dataset (32 tasks: 8 prompts x 4 models)."""
    if dataset not in ("asap", "toefl11"):
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}. Use 'asap' or 'toefl11'.")

    lock = FileLock(str(TASKS_DIR / f".init_{dataset}.lock"))

    with lock:
        # Filter existing tasks for this dataset
        prefix = f"{dataset}_" if dataset == "toefl11" else ""
        existing = [f for f in TASKS_DIR.glob("*.json")
                   if (dataset == "toefl11" and f.stem.startswith("toefl11_")) or
                      (dataset == "asap" and not f.stem.startswith("toefl11_"))]

        if existing and not force:
            return {
                "message": f"{dataset.upper()} tasks already exist ({len(existing)} tasks). Use force=true to reinitialize.",
                "initialized": False,
            }

        # Clear existing tasks for this dataset if force
        if force:
            for f in existing:
                f.unlink()

        # Create tasks for all prompt x model combinations
        created = 0
        for prompt_id in range(1, 9):
            for model_name, model_short in MODELS:
                if dataset == "toefl11":
                    task_id = f"toefl11_prompt{prompt_id}_{model_short}"
                else:
                    task_id = f"prompt{prompt_id}_{model_short}"
                task = TaskInfo(
                    task_id=task_id,
                    prompt_id=prompt_id,
                    model_name=model_name,
                    model_short_name=model_short,
                    status=TaskStatus.PENDING,
                    dataset=dataset,
                    max_epochs=30,
                )
                save_task(task)
                created += 1

        logger.info(f"Initialized {created} {dataset.upper()} tasks")
        return {
            "message": f"Initialized {created} {dataset.upper()} tasks",
            "initialized": True,
            "task_count": created,
            "dataset": dataset,
        }


@app.get("/tasks/next", response_model=TaskStartResponse)
async def get_next_task(
    worker_id: str = Query(..., description="Worker identifier"),
    dataset: str = Query(None, description="Filter by dataset (asap or toefl11). If None, returns any pending task.")
):
    """Get next available task for a worker."""
    lock = FileLock(str(TASKS_DIR / ".assign.lock"))

    with lock:
        tasks = get_all_tasks()

        # Find first pending task (optionally filtered by dataset)
        for task in tasks:
            # Filter by dataset if specified
            if dataset is not None:
                is_toefl = task.task_id.startswith("toefl11_")
                if dataset == "toefl11" and not is_toefl:
                    continue
                if dataset == "asap" and is_toefl:
                    continue

            if task.status == TaskStatus.PENDING:
                # Assign to worker
                task.status = TaskStatus.RUNNING
                task.worker_id = worker_id
                task.started_at = datetime.now()
                save_task(task)

                logger.info(f"Assigned task {task.task_id} to worker {worker_id}")
                return TaskStartResponse(
                    success=True,
                    task=task,
                    message=f"Task {task.task_id} assigned",
                )

        # No pending tasks
        return TaskStartResponse(
            success=False,
            message="No pending tasks available",
        )


@app.get("/tasks", response_model=StatusResponse)
async def get_status():
    """Get overall status of all tasks with verified epochs."""
    tasks = get_all_tasks()

    # Update each task with verified epoch
    for task in tasks:
        verified_epoch = get_verified_last_completed_epoch(task.task_id)
        task.last_completed_epoch = verified_epoch

    status_counts = {s: 0 for s in TaskStatus}
    for task in tasks:
        status_counts[task.status] += 1

    return StatusResponse(
        total_tasks=len(tasks),
        pending=status_counts[TaskStatus.PENDING],
        running=status_counts[TaskStatus.RUNNING],
        completed=status_counts[TaskStatus.COMPLETED],
        failed=status_counts[TaskStatus.FAILED],
        tasks=tasks,
    )


@app.get("/tasks/matrix")
async def get_matrix(dataset: str = Query("asap", description="Dataset to show (asap or toefl11)")):
    """Get task status as a matrix (prompts x models) with verified epochs."""
    if dataset not in ("asap", "toefl11"):
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")

    tasks = {t.task_id: t for t in get_all_tasks()}

    status_chars = {
        TaskStatus.PENDING: "P",
        TaskStatus.RUNNING: "R",
        TaskStatus.COMPLETED: "C",
        TaskStatus.FAILED: "F",
    }

    matrix = {}
    for prompt_id in range(1, 9):
        row = {}
        for _, model_short in MODELS:
            if dataset == "toefl11":
                task_id = f"toefl11_prompt{prompt_id}_{model_short}"
            else:
                task_id = f"prompt{prompt_id}_{model_short}"
            task = tasks.get(task_id)
            if task:
                # Use verified epoch instead of stored epoch
                verified_epoch = get_verified_last_completed_epoch(task_id)
                row[model_short] = {
                    "status": status_chars.get(task.status, "?"),
                    "epoch": verified_epoch,
                    "worker": task.worker_id,
                }
            else:
                row[model_short] = {"status": "-", "epoch": 0}
        matrix[f"prompt{prompt_id}"] = row

    return {"dataset": dataset, "matrix": matrix}


@app.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    """Get task information with verified last_completed_epoch."""
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Override with verified epoch (where both checkpoint AND metrics exist)
    verified_epoch = get_verified_last_completed_epoch(task_id)
    if verified_epoch != task.last_completed_epoch:
        logger.info(f"Task {task_id}: stored epoch={task.last_completed_epoch}, verified epoch={verified_epoch}")
        task.last_completed_epoch = verified_epoch

    return task


@app.get("/tasks/{task_id}/config", response_model=ExperimentConfig)
async def get_task_config(task_id: str):
    """Get experiment configuration for a task."""
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return ExperimentConfig(
        task_id=task_id,
        prompt_id=task.prompt_id,
        model_name=task.model_name,
        dataset=task.dataset,
        max_epochs=task.max_epochs,
    )


@app.post("/tasks/{task_id}/progress")
async def update_progress(task_id: str, epoch: int = Query(...), worker_id: str = Query(...)):
    """Update task progress (last completed epoch).

    Only accepts progress update if both checkpoint and metrics exist for the epoch.
    """
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.worker_id != worker_id:
        raise HTTPException(status_code=403, detail="Worker ID mismatch")

    # Verify that metrics exist for this epoch (required for epoch to be complete)
    metrics_file = RESULTS_DIR / task_id / f"metrics_epoch_{epoch}.json"
    if not metrics_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Cannot mark epoch {epoch} as complete: metrics not uploaded yet"
        )

    # For epoch > 0, also verify checkpoint exists (as zip or directory)
    if epoch > 0:
        checkpoint_epoch_dir = CHECKPOINTS_DIR / task_id / f"epoch_{epoch}"
        checkpoint_exists = (
            (checkpoint_epoch_dir / "adapter.zip").exists() or
            (checkpoint_epoch_dir / "adapter").is_dir()
        )
        if not checkpoint_exists:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot mark epoch {epoch} as complete: checkpoint not uploaded yet"
            )

    task.last_completed_epoch = epoch
    save_task(task)

    logger.info(f"Task {task_id}: epoch {epoch} completed (verified)")
    return {"message": f"Progress updated: epoch {epoch}"}


@app.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str, request: TaskCompleteRequest):
    """Mark task as completed."""
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.worker_id != request.worker_id:
        raise HTTPException(status_code=403, detail="Worker ID mismatch")

    task.status = TaskStatus.COMPLETED
    task.completed_at = datetime.now()
    save_task(task)

    # Save summary
    summary_path = RESULTS_DIR / task_id / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(request.summary, f, indent=2)

    logger.info(f"Task {task_id} completed by {request.worker_id}")
    return {"message": f"Task {task_id} completed"}


@app.post("/tasks/{task_id}/fail")
async def fail_task(task_id: str, request: TaskFailRequest):
    """Mark task as failed."""
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task.status = TaskStatus.FAILED
    task.error_message = request.error_message
    save_task(task)

    logger.warning(f"Task {task_id} failed: {request.error_message}")
    return {"message": f"Task {task_id} marked as failed"}


@app.post("/tasks/{task_id}/reset")
async def reset_task(task_id: str):
    """Reset a task to pending status."""
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task.status = TaskStatus.PENDING
    task.worker_id = None
    task.started_at = None
    task.completed_at = None
    task.error_message = None
    # Keep last_completed_epoch for resume capability
    save_task(task)

    logger.info(f"Task {task_id} reset to pending")
    return {"message": f"Task {task_id} reset"}


class TaskReleaseRequest(BaseModel):
    worker_id: str


@app.post("/tasks/{task_id}/release")
async def release_task(task_id: str, request: TaskReleaseRequest):
    """
    Release a task back to pending status (for graceful worker shutdown).

    Unlike reset, this preserves progress and only releases if the
    requesting worker owns the task.
    """
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    # Only allow release if worker owns the task
    if task.worker_id != request.worker_id:
        raise HTTPException(
            status_code=403,
            detail=f"Task {task_id} is not owned by worker {request.worker_id}"
        )

    if task.status != TaskStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} is not running (status: {task.status})"
        )

    # Release task back to pending, preserving progress
    task.status = TaskStatus.PENDING
    task.worker_id = None
    task.started_at = None
    # Keep last_completed_epoch for resume capability
    save_task(task)

    logger.info(f"Task {task_id} released by worker {request.worker_id}")
    return {"message": f"Task {task_id} released"}


# ============================================
# Checkpoint Endpoints
# ============================================

@app.post("/checkpoints/{task_id}/epoch/{epoch}")
async def upload_checkpoint(task_id: str, epoch: int, file: UploadFile = File(...)):
    """Upload a checkpoint (adapter) as a zip file."""
    checkpoint_dir = CHECKPOINTS_DIR / task_id / f"epoch_{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    zip_path = checkpoint_dir / "adapter.zip"

    async with aiofiles.open(zip_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    logger.info(f"Checkpoint saved: {task_id}/epoch_{epoch} ({len(content)} bytes)")
    return {"message": f"Checkpoint uploaded", "size_bytes": len(content)}


@app.get("/checkpoints/{task_id}/epoch/{epoch}")
async def download_checkpoint(task_id: str, epoch: int):
    """Download a checkpoint (adapter) zip file."""
    zip_path = CHECKPOINTS_DIR / task_id / f"epoch_{epoch}" / "adapter.zip"

    if not zip_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {task_id}/epoch_{epoch}")

    return FileResponse(
        path=zip_path,
        filename=f"{task_id}_epoch_{epoch}_adapter.zip",
        media_type="application/zip",
    )


@app.get("/checkpoints/{task_id}/epoch/{epoch}/exists", response_model=CheckpointInfo)
async def check_checkpoint(task_id: str, epoch: int):
    """Check if a checkpoint exists."""
    zip_path = CHECKPOINTS_DIR / task_id / f"epoch_{epoch}" / "adapter.zip"

    exists = zip_path.exists()
    size = zip_path.stat().st_size if exists else None

    return CheckpointInfo(
        task_id=task_id,
        epoch=epoch,
        exists=exists,
        size_bytes=size,
    )


@app.get("/checkpoints/{task_id}/last")
async def get_last_checkpoint(task_id: str):
    """Get the last available checkpoint epoch for a task."""
    task_dir = CHECKPOINTS_DIR / task_id
    if not task_dir.exists():
        return {"last_epoch": 0}

    last_epoch = 0
    for epoch_dir in task_dir.glob("epoch_*"):
        if (epoch_dir / "adapter.zip").exists():
            epoch = int(epoch_dir.name.split("_")[1])
            last_epoch = max(last_epoch, epoch)

    return {"last_epoch": last_epoch}


# ============================================
# Results Endpoints
# ============================================

@app.post("/results/{task_id}/epoch/{epoch}/metrics")
async def upload_metrics(task_id: str, epoch: int, metrics: dict):
    """Upload metrics for an epoch."""
    results_dir = RESULTS_DIR / task_id
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / f"metrics_epoch_{epoch}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved: {task_id}/epoch_{epoch}")
    return {"message": "Metrics uploaded"}


@app.get("/results/{task_id}/epoch/{epoch}/metrics")
async def download_metrics(task_id: str, epoch: int):
    """Download metrics for an epoch."""
    metrics_path = RESULTS_DIR / task_id / f"metrics_epoch_{epoch}.json"

    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"Metrics not found: {task_id}/epoch_{epoch}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics


@app.post("/results/{task_id}/epoch/{epoch}/predictions")
async def upload_predictions(task_id: str, epoch: int, file: UploadFile = File(...)):
    """Upload predictions CSV for an epoch."""
    results_dir = RESULTS_DIR / task_id
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = results_dir / f"predictions_epoch_{epoch}.csv"

    async with aiofiles.open(predictions_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    logger.info(f"Predictions saved: {task_id}/epoch_{epoch}")
    return {"message": "Predictions uploaded"}


@app.get("/results/{task_id}/epoch/{epoch}/predictions")
async def download_predictions(task_id: str, epoch: int):
    """Download predictions CSV for an epoch."""
    predictions_path = RESULTS_DIR / task_id / f"predictions_epoch_{epoch}.csv"

    if not predictions_path.exists():
        raise HTTPException(status_code=404, detail=f"Predictions not found: {task_id}/epoch_{epoch}")

    return FileResponse(
        path=predictions_path,
        filename=f"{task_id}_predictions_epoch_{epoch}.csv",
        media_type="text/csv",
    )


@app.get("/results/{task_id}/epoch/{epoch}/exists")
async def check_results(task_id: str, epoch: int):
    """Check if results exist for an epoch."""
    results_dir = RESULTS_DIR / task_id
    metrics_exists = (results_dir / f"metrics_epoch_{epoch}.json").exists()
    predictions_exists = (results_dir / f"predictions_epoch_{epoch}.csv").exists()

    return {
        "metrics_exists": metrics_exists,
        "predictions_exists": predictions_exists,
        "complete": metrics_exists and predictions_exists,
    }


@app.get("/results/{task_id}/last")
async def get_last_results(task_id: str):
    """Get the last epoch with complete results."""
    results_dir = RESULTS_DIR / task_id
    if not results_dir.exists():
        return {"last_epoch": -1}  # -1 means epoch 0 not done yet

    last_epoch = -1
    for epoch in range(0, 100):  # Check up to 100 epochs
        metrics_path = results_dir / f"metrics_epoch_{epoch}.json"
        if metrics_path.exists():
            last_epoch = epoch
        else:
            break

    return {"last_epoch": last_epoch}


# ============================================
# Data Endpoints
# ============================================

@app.get("/data/asap")
async def get_asap_data():
    """Download ASAP dataset."""
    data_path = ASAP_DATA_DIR / "training_set_rel3.tsv"

    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail="ASAP data not found. Please place training_set_rel3.tsv in server/data/asap/"
        )

    return FileResponse(
        path=data_path,
        filename="training_set_rel3.tsv",
        media_type="text/tab-separated-values",
    )


@app.get("/data/asap/exists")
async def check_asap_data():
    """Check if ASAP data exists on server."""
    data_path = ASAP_DATA_DIR / "training_set_rel3.tsv"
    return {"exists": data_path.exists()}


# ============================================
# TOEFL11 Data Endpoints
# ============================================

@app.get("/data/toefl11/index")
async def get_toefl11_index():
    """Download TOEFL11 index.csv."""
    index_path = TOEFL11_DATA_DIR / "data" / "text" / "index.csv"

    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="TOEFL11 index.csv not found."
        )

    return FileResponse(
        path=index_path,
        filename="index.csv",
        media_type="text/csv",
    )


@app.get("/data/toefl11/essay/{filename}")
async def get_toefl11_essay(filename: str):
    """Download a TOEFL11 essay file."""
    essay_path = TOEFL11_DATA_DIR / "data" / "text" / "responses" / "original" / filename

    if not essay_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Essay not found: {filename}"
        )

    return FileResponse(
        path=essay_path,
        filename=filename,
        media_type="text/plain",
    )


@app.get("/data/toefl11/prompt/{prompt_id}")
async def get_toefl11_prompt(prompt_id: int):
    """Download a TOEFL11 prompt file (P1-P8)."""
    prompt_path = TOEFL11_DATA_DIR / "data" / "text" / "prompts" / f"P{prompt_id}.txt"

    if not prompt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prompt not found: P{prompt_id}"
        )

    return FileResponse(
        path=prompt_path,
        filename=f"P{prompt_id}.txt",
        media_type="text/plain",
    )


@app.get("/data/toefl11/rubric")
async def get_toefl11_rubric():
    """Download TOEFL11 rubric (Band format)."""
    rubric_path = DATA_DIR / "TOEFL11" / "rubric_bands.md"

    if not rubric_path.exists():
        raise HTTPException(
            status_code=404,
            detail="TOEFL11 rubric not found."
        )

    return FileResponse(
        path=rubric_path,
        filename="rubric_bands.md",
        media_type="text/markdown",
    )


@app.get("/data/toefl11/exists")
async def check_toefl11_data():
    """Check if TOEFL11 data exists on server."""
    index_path = TOEFL11_DATA_DIR / "data" / "text" / "index.csv"
    rubric_path = DATA_DIR / "TOEFL11" / "rubric_bands.md"
    prompts_dir = TOEFL11_DATA_DIR / "data" / "text" / "prompts"

    return {
        "index_exists": index_path.exists(),
        "rubric_exists": rubric_path.exists(),
        "prompts_exist": prompts_dir.exists() and len(list(prompts_dir.glob("P*.txt"))) == 8,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def run_server():
    """Run the server."""
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
