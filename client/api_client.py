"""
API client for communicating with the experiment server.
"""

import io
import os
import json
import zipfile
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)


class APIClient:
    """Client for communicating with the experiment server."""

    def __init__(self, server_url: str, worker_id: str):
        """
        Initialize API client.

        Args:
            server_url: Base URL of the server (e.g., "http://192.168.100.10:8000")
            worker_id: Unique identifier for this worker
        """
        self.server_url = server_url.rstrip('/')
        self.worker_id = worker_id
        self.timeout = 300  # 5 minutes for large file uploads

    def _url(self, path: str) -> str:
        """Build full URL."""
        return f"{self.server_url}{path}"

    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            resp = requests.get(self._url("/health"), timeout=10)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ============================================
    # Task Management
    # ============================================

    def get_next_task(self, dataset: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get next available task.

        Args:
            dataset: Filter by dataset ("asap" or "toefl11"). If None, returns any pending task.
        """
        try:
            params = {"worker_id": self.worker_id}
            if dataset is not None:
                params["dataset"] = dataset
            resp = requests.get(
                self._url("/tasks/next"),
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("success"):
                return data.get("task")
            return None

        except requests.RequestException as e:
            logger.error(f"Failed to get next task: {e}")
            return None

    def get_task_config(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment configuration for a task."""
        try:
            resp = requests.get(self._url(f"/tasks/{task_id}/config"), timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get task config: {e}")
            return None

    def update_progress(self, task_id: str, epoch: int) -> bool:
        """Update task progress."""
        try:
            resp = requests.post(
                self._url(f"/tasks/{task_id}/progress"),
                params={"epoch": epoch, "worker_id": self.worker_id},
                timeout=30,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to update progress: {e}")
            return False

    def complete_task(self, task_id: str, summary: Dict[str, Any]) -> bool:
        """Mark task as completed."""
        try:
            resp = requests.post(
                self._url(f"/tasks/{task_id}/complete"),
                json={"worker_id": self.worker_id, "summary": summary},
                timeout=30,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to complete task: {e}")
            return False

    def fail_task(self, task_id: str, error_message: str) -> bool:
        """Mark task as failed."""
        try:
            resp = requests.post(
                self._url(f"/tasks/{task_id}/fail"),
                json={"worker_id": self.worker_id, "error_message": error_message},
                timeout=30,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to mark task as failed: {e}")
            return False

    def release_task(self, task_id: str) -> bool:
        """
        Release task back to pending status (for graceful shutdown).

        This allows another worker to pick up the task while preserving progress.
        """
        try:
            resp = requests.post(
                self._url(f"/tasks/{task_id}/release"),
                json={"worker_id": self.worker_id},
                timeout=30,
            )
            resp.raise_for_status()
            logger.info(f"Released task: {task_id}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to release task: {e}")
            return False

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get overall status."""
        try:
            resp = requests.get(self._url("/tasks"), timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get status: {e}")
            return None

    # ============================================
    # Checkpoint Management
    # ============================================

    def upload_checkpoint(self, task_id: str, epoch: int, adapter_dir: Path) -> bool:
        """
        Upload a checkpoint (adapter directory) as a zip file.

        Args:
            task_id: Task identifier
            epoch: Epoch number
            adapter_dir: Path to adapter directory

        Returns:
            True if successful
        """
        try:
            # Create zip file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in adapter_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(adapter_dir)
                        zf.write(file_path, arcname)

            zip_buffer.seek(0)

            resp = requests.post(
                self._url(f"/checkpoints/{task_id}/epoch/{epoch}"),
                files={"file": ("adapter.zip", zip_buffer, "application/zip")},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            logger.info(f"Uploaded checkpoint: {task_id}/epoch_{epoch}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to upload checkpoint: {e}")
            return False

    def download_checkpoint(self, task_id: str, epoch: int, target_dir: Path) -> bool:
        """
        Download a checkpoint and extract to target directory.

        Args:
            task_id: Task identifier
            epoch: Epoch number
            target_dir: Directory to extract adapter to

        Returns:
            True if successful
        """
        try:
            resp = requests.get(
                self._url(f"/checkpoints/{task_id}/epoch/{epoch}"),
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()

            # Extract to target directory
            target_dir.mkdir(parents=True, exist_ok=True)

            zip_buffer = io.BytesIO(resp.content)
            with zipfile.ZipFile(zip_buffer, 'r') as zf:
                zf.extractall(target_dir)

            logger.info(f"Downloaded checkpoint: {task_id}/epoch_{epoch}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to download checkpoint: {e}")
            return False

    def check_checkpoint(self, task_id: str, epoch: int) -> bool:
        """Check if checkpoint exists on server."""
        try:
            resp = requests.get(
                self._url(f"/checkpoints/{task_id}/epoch/{epoch}/exists"),
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("exists", False)
        except requests.RequestException:
            return False

    def get_last_checkpoint(self, task_id: str) -> int:
        """Get the last checkpoint epoch on server."""
        try:
            resp = requests.get(
                self._url(f"/checkpoints/{task_id}/last"),
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("last_epoch", 0)
        except requests.RequestException:
            return 0

    # ============================================
    # Results Management
    # ============================================

    def upload_metrics(self, task_id: str, epoch: int, metrics: Dict[str, Any]) -> bool:
        """Upload metrics for an epoch."""
        import math
        import json

        def replace_nan(obj):
            """Recursively replace NaN/Inf values with None."""
            if isinstance(obj, dict):
                return {k: replace_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan(v) for v in obj]
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj

        try:
            # Replace NaN/Inf values before JSON serialization
            clean_metrics = replace_nan(metrics)
            resp = requests.post(
                self._url(f"/results/{task_id}/epoch/{epoch}/metrics"),
                json=clean_metrics,
                timeout=30,
            )
            resp.raise_for_status()
            logger.info(f"Uploaded metrics: {task_id}/epoch_{epoch}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to upload metrics: {e}")
            return False

    def download_metrics(self, task_id: str, epoch: int) -> Optional[Dict[str, Any]]:
        """Download metrics for an epoch."""
        try:
            resp = requests.get(
                self._url(f"/results/{task_id}/epoch/{epoch}/metrics"),
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def upload_predictions(self, task_id: str, epoch: int, predictions_path: Path) -> bool:
        """Upload predictions CSV for an epoch."""
        try:
            with open(predictions_path, 'rb') as f:
                resp = requests.post(
                    self._url(f"/results/{task_id}/epoch/{epoch}/predictions"),
                    files={"file": (predictions_path.name, f, "text/csv")},
                    timeout=self.timeout,
                )
            resp.raise_for_status()
            logger.info(f"Uploaded predictions: {task_id}/epoch_{epoch}")
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to upload predictions: {e}")
            return False

    def check_results(self, task_id: str, epoch: int) -> Dict[str, bool]:
        """Check if results exist for an epoch."""
        try:
            resp = requests.get(
                self._url(f"/results/{task_id}/epoch/{epoch}/exists"),
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {"metrics_exists": False, "predictions_exists": False, "complete": False}

    def get_last_results(self, task_id: str) -> int:
        """Get last epoch with complete results. Returns -1 if epoch 0 not done."""
        try:
            resp = requests.get(
                self._url(f"/results/{task_id}/last"),
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("last_epoch", -1)
        except requests.RequestException:
            return -1

    # ============================================
    # Data Management
    # ============================================

    def download_asap_data(self, target_path: Path) -> bool:
        """Download ASAP dataset from server."""
        try:
            resp = requests.get(
                self._url("/data/asap"),
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded ASAP data to {target_path}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to download ASAP data: {e}")
            return False

    def check_asap_data(self) -> bool:
        """Check if ASAP data exists on server."""
        try:
            resp = requests.get(self._url("/data/asap/exists"), timeout=30)
            resp.raise_for_status()
            return resp.json().get("exists", False)
        except requests.RequestException:
            return False

    def download_toefl11_data(self, target_dir: Path) -> bool:
        """
        Download TOEFL11 dataset from server.

        Downloads index.csv and all essay files from the server.

        Args:
            target_dir: Directory to save the TOEFL11 data (ETS_Corpus_of_Non-Native_Written_English)

        Returns:
            True if successful
        """
        import pandas as pd

        try:
            # Create directory structure
            text_dir = target_dir / "data" / "text"
            responses_dir = text_dir / "responses" / "original"
            prompts_dir = text_dir / "prompts"

            responses_dir.mkdir(parents=True, exist_ok=True)
            prompts_dir.mkdir(parents=True, exist_ok=True)

            # Download index.csv
            logger.info("Downloading TOEFL11 index.csv...")
            resp = requests.get(
                self._url("/data/toefl11/index"),
                timeout=self.timeout,
            )
            resp.raise_for_status()

            index_path = text_dir / "index.csv"
            with open(index_path, 'wb') as f:
                f.write(resp.content)

            # Parse index to get list of essay files
            df = pd.read_csv(index_path)
            filenames = df['Filename'].tolist()

            # Download all essay files
            logger.info(f"Downloading {len(filenames)} TOEFL11 essays...")
            for i, filename in enumerate(filenames):
                if i % 500 == 0:
                    logger.info(f"  Progress: {i}/{len(filenames)}")

                resp = requests.get(
                    self._url(f"/data/toefl11/essay/{filename}"),
                    timeout=60,
                )
                resp.raise_for_status()

                essay_path = responses_dir / filename
                with open(essay_path, 'wb') as f:
                    f.write(resp.content)

            # Download prompts P1-P8
            logger.info("Downloading TOEFL11 prompts...")
            for prompt_id in range(1, 9):
                resp = requests.get(
                    self._url(f"/data/toefl11/prompt/{prompt_id}"),
                    timeout=30,
                )
                resp.raise_for_status()

                prompt_path = prompts_dir / f"P{prompt_id}.txt"
                with open(prompt_path, 'wb') as f:
                    f.write(resp.content)

            logger.info(f"Downloaded TOEFL11 data to {target_dir}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to download TOEFL11 data: {e}")
            return False
        except Exception as e:
            logger.error(f"Error processing TOEFL11 data: {e}")
            return False

    def check_toefl11_data(self) -> Dict[str, bool]:
        """Check if TOEFL11 data exists on server."""
        try:
            resp = requests.get(self._url("/data/toefl11/exists"), timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {"index_exists": False, "rubric_exists": False, "prompts_exist": False}
