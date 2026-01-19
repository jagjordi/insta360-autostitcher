#!/usr/bin/env python3
"""Controller service for Insta360 auto-stitching.

This module keeps track of pending RAW files using a SQLite database, exposes
REST endpoints to orchestrate scanning and stitching, and executes the
MediaSDKTest CLI when work needs to be done.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Iterable, List, Optional, Set
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, jsonify, request, send_from_directory

APP_STORAGE_DIR = os.getenv("APP_STORAGE_DIR", "/app").rstrip("/")
RAW_DIR = os.path.join(APP_STORAGE_DIR, "raw")
OUT_DIR = os.path.join(APP_STORAGE_DIR, "stitched")
DATABASE_PATH = os.getenv(
    "AUTO_STITCHER_DB", os.path.join(APP_STORAGE_DIR, "autostitcher.db")
)
THUMBNAIL_DIR = os.path.join(APP_STORAGE_DIR, "thumbnails")
DEFAULT_STITCH_CONCURRENCY = max(
    int(os.getenv("STITCH_CONCURRENCY", os.getenv("MAX_PARALLEL_JOBS", "1"))), 1
)
DEFAULT_SCAN_CONCURRENCY = max(int(os.getenv("SCAN_CONCURRENCY", "25")), 1)
DEFAULT_DEEP_SCAN_CONCURRENCY = max(int(os.getenv("DEEP_SCAN_CONCURRENCY", "4")), 1)
DEFAULT_THUMBNAIL_CONCURRENCY = max(int(os.getenv("THUMBNAIL_CONCURRENCY", "4")), 1)
try:
    DEFAULT_EXPECTED_RATIO = float(os.getenv("EXPECTED_SIZE_RATIO", "1.0"))
except ValueError:
    DEFAULT_EXPECTED_RATIO = 1.0
if DEFAULT_EXPECTED_RATIO <= 0:
    DEFAULT_EXPECTED_RATIO = 1.0
DEFAULT_OUTPUT_SIZE = os.getenv("OUTPUT_SIZE", "5760x2880")
DEFAULT_BITRATE = os.getenv("BITRATE", "200000000")
DEFAULT_ORIGINAL_BITRATE = os.getenv("ORIGINAL_BITRATE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_STITCH_TYPE = os.getenv("STITCH_TYPE", "dynamicstitch")
DEFAULT_AUTO_RESOLUTION = os.getenv("AUTO_RESOLUTION", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MIN_SUCCESS_RATIO = 0.5
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "600"))
REST_PORT = int(os.getenv("AUTO_STITCHER_PORT", "8000"))
LOG_LEVEL = os.getenv("AUTO_STITCHER_LOG_LEVEL", "INFO")
DEBUG_FLAG = os.getenv("AUTO_STITCHER_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
LOGIN_TOKEN = os.getenv("LOGIN_TOKEN", "")

STATUS_UNPROCESSED = "unprocessed"
STATUS_PROCESSING = "processing"
STATUS_PROCESSED = "processed"
STATUS_FAILED = "failed"
STATUS_VALUES = {
    STATUS_UNPROCESSED,
    STATUS_PROCESSING,
    STATUS_PROCESSED,
    STATUS_FAILED,
}

FILE_PATTERN = re.compile(r"VID_(\d{8}_\d{6})_00_(\d{3})\.insv")
LOGGER = logging.getLogger("insta360_autostitcher")


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    LOGGER.setLevel(level)


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def format_creation_time(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp).isoformat()
    except ValueError:
        pass
    try:
        return datetime.strptime(timestamp, "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        pass
    if " " in timestamp:
        try:
            return datetime.fromisoformat(timestamp.replace(" ", "T")).isoformat()
        except ValueError:
            pass
    return timestamp


def inject_spherical_metadata(path: str) -> bool:
    temp_output = f"{path}.spatial.mp4"
    cmd = [
        "spatialmedia",
        "-i",
        path,
        temp_output,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)  # noqa: S603
    except FileNotFoundError:
        LOGGER.warning("spatialmedia not found; skipping spherical metadata for %s", path)
        return False
    if result.returncode != 0:
        LOGGER.warning(
            "spatialmedia metadata injection failed for %s: %s",
            path,
            result.stderr.decode(errors="ignore"),
        )
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except OSError:
                LOGGER.warning("Failed to remove spatialmedia temp output %s", temp_output)
        return False
    try:
        os.replace(temp_output, path)
    except OSError:
        LOGGER.warning("Failed to replace %s with spatialmedia output %s", path, temp_output)
        return False
    return True


def ensure_dirs() -> None:
    LOGGER.debug(
        "Ensuring storage at %s with RAW=%s OUT=%s THUMBNAILS=%s",
        APP_STORAGE_DIR,
        RAW_DIR,
        OUT_DIR,
        THUMBNAIL_DIR,
    )
    os.makedirs(APP_STORAGE_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)


def is_file_writing(path: str) -> bool:
    if not os.path.exists(path):
        return True
    size = os.path.getsize(path)
    time.sleep(1)
    writing = not os.path.exists(path) or os.path.getsize(path) != size
    if writing:
        LOGGER.debug("File %s still writing (size changed)", path)
    return writing


def stitched_path(timestamp: str) -> str:
    return os.path.join(OUT_DIR, f"VID_{timestamp}.mp4")


def log_path(timestamp: str) -> str:
    return os.path.join(OUT_DIR, f"VID_{timestamp}.log")


def thumbnail_path(job_id: str) -> str:
    return os.path.join(THUMBNAIL_DIR, f"{job_id}.jpg")


def configure_paths(
    storage_dir: Optional[str] = None,
    raw_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Update global storage paths from CLI args or environment."""
    global APP_STORAGE_DIR, RAW_DIR, OUT_DIR, DATABASE_PATH
    if storage_dir:
        APP_STORAGE_DIR = storage_dir.rstrip("/")
    RAW_DIR = raw_dir or os.path.join(APP_STORAGE_DIR, "raw")
    OUT_DIR = out_dir or os.path.join(APP_STORAGE_DIR, "stitched")
    env_db = os.getenv("AUTO_STITCHER_DB")
    DATABASE_PATH = db_path or env_db or os.path.join(APP_STORAGE_DIR, "autostitcher.db")


def count_video_streams(path: str) -> Optional[int]:
    """Return the number of video streams in the given file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False  # noqa: S603
        )
    except FileNotFoundError:
        LOGGER.error("ffprobe is not installed or not found in PATH; skipping %s", path)
        return None
    if result.returncode != 0:
        LOGGER.warning("ffprobe failed for %s: %s", path, result.stderr.strip())
        return None
    streams = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return len(streams)


def video_resolution(path: str) -> Optional[tuple[int, int]]:
    """Return (width, height) for the primary video stream."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False  # noqa: S603
        )
    except FileNotFoundError:
        LOGGER.error("ffprobe not found when probing resolution for %s", path)
        return None
    if result.returncode != 0:
        LOGGER.warning("Unable to probe resolution for %s: %s", path, result.stderr.strip())
        return None
    data = result.stdout.strip().split(",")
    if not data or not data[0]:
        return None
    try:
        if len(data) == 1 and "x" in data[0]:
            width_str, height_str = data[0].split("x", maxsplit=1)
        elif len(data) >= 2:
            width_str, height_str = data[0], data[1]
        else:
            return None
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            return None
        return width, height
    except (ValueError, IndexError):
        LOGGER.warning("Invalid resolution output for %s: %s", path, result.stdout.strip())
        return None


def video_duration_seconds(path: str) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False  # noqa: S603
        )
    except FileNotFoundError:
        LOGGER.error("ffprobe not found when probing %s", path)
        return None
    if result.returncode != 0:
        LOGGER.warning("Unable to probe duration for %s: %s", path, result.stderr.strip())
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        LOGGER.warning("Invalid duration output for %s: %s", path, result.stdout.strip())
        return None


def extract_thumbnail(source: str, output: str, timestamp: float) -> bool:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    seek = max(timestamp, 0)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(seek),
        "-i",
        source,
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-vf",
        "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2",
        "-q:v",
        "2",
        output,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)  # noqa: S603
    except FileNotFoundError:
        LOGGER.error("ffmpeg not found when generating thumbnail for %s", source)
        return False
    if result.returncode != 0:
        LOGGER.warning(
            "ffmpeg failed for %s: %s", source, result.stderr.decode(errors="ignore")
        )
        return False
    return True


@dataclass
class JobCandidate:
    timestamp: str
    segment: str
    source_files: List[str]

    @property
    def final_file(self) -> str:
        return stitched_path(self.timestamp)

    def sources(self) -> List[str]:
        return list(self.source_files)


@dataclass
class ActiveTask:
    task_id: str
    action: str
    started_at: str
    cancel_event: threading.Event
    thread: threading.Thread


class JobDatabase:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _locked(self) -> Iterable[None]:
        acquired = self._lock.acquire(timeout=60)
        if not acquired:
            raise TimeoutError("Timed out waiting for database lock")
        try:
            yield
        finally:
            self._lock.release()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=60)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._locked():
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        final_file TEXT NOT NULL UNIQUE,
                        source_files TEXT NOT NULL,
                        status TEXT NOT NULL,
                        pid INTEGER,
                        stitched_size INTEGER DEFAULT 0,
                        process REAL DEFAULT 0,
                        expected_size INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_jobs_status
                    ON jobs(status)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                conn.commit()

    def insert_job(self, *, timestamp: str, final_file: str, source_files: List[str],
                   status: str, expected_size: int) -> str:
        job_id = str(uuid.uuid4())
        payload = json.dumps(source_files)
        now = utc_now()
        with self._locked():
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO jobs(id, timestamp, final_file, source_files, status,
                                     expected_size, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, timestamp, final_file, payload, status, expected_size, now, now),
                )
                conn.commit()
        return job_id

    def update_job(self, job_id: str, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in fields.keys())
        values = list(fields.values()) + [job_id]
        with self._locked():
            with closing(self._connect()) as conn:
                conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", values)
                conn.commit()

    def fetch_jobs(self, statuses: Optional[Iterable[str]] = None) -> List[sqlite3.Row]:
        with self._locked():
            with closing(self._connect()) as conn:
                if statuses:
                    placeholders = ",".join("?" for _ in statuses)
                    query = f"SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY timestamp"
                    rows = conn.execute(query, tuple(statuses)).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM jobs ORDER BY timestamp").fetchall()
        return rows

    def fetch_job(self, job_id: str) -> Optional[sqlite3.Row]:
        with self._locked():
            with closing(self._connect()) as conn:
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row

    def get_setting(self, key: str) -> Optional[str]:
        with self._locked():
            with closing(self._connect()) as conn:
                row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self._locked():
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO settings(key, value)
                    VALUES(?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value
                    """,
                    (key, value),
                )
                conn.commit()

    def fetch_jobs_by_ids(self, job_ids: Iterable[str]) -> List[sqlite3.Row]:
        job_ids = list(job_ids)
        if not job_ids:
            return []
        placeholders = ",".join("?" for _ in job_ids)
        with self._locked():
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    f"SELECT * FROM jobs WHERE id IN ({placeholders})", job_ids
                ).fetchall()
        return rows

    def find_by_final_file(self, final_file: str) -> Optional[sqlite3.Row]:
        with self._locked():
            with closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE final_file = ?", (final_file,)
                ).fetchone()
        return row

    def list_final_files(self) -> Set[str]:
        with self._locked():
            with closing(self._connect()) as conn:
                rows = conn.execute("SELECT final_file FROM jobs").fetchall()
        return {row[0] for row in rows if row[0]}


class AutoStitcher:
    def __init__(self, debug: bool = False) -> None:
        ensure_dirs()
        self.db = JobDatabase(DATABASE_PATH)
        self._task_lock = threading.Lock()
        self._active_tasks: Dict[str, ActiveTask] = {}
        self._lock = threading.Lock()
        self._active_threads: Dict[str, threading.Thread] = {}
        self._pending_jobs: Deque[str] = deque()
        self._pending_job_ids: Set[str] = set()
        self._job_temp_outputs: Dict[str, str] = {}
        self.debug_mode = debug
        (
            self.stitch_parallelism,
            self.scan_parallelism,
            self.deep_scan_parallelism,
            self.thumbnail_parallelism,
        ) = self._load_concurrency_settings()
        self.expected_size_ratio = self._load_expected_ratio()
        (
            self.output_size,
            self.bitrate,
            self.stitch_type,
            self.auto_resolution,
            self.original_bitrate,
        ) = self._load_stitch_settings()
        LOGGER.info(
            "AutoStitcher initialized with DB at %s (debug=%s)", DATABASE_PATH, self.debug_mode
        )

    def _load_concurrency_settings(self) -> tuple[int, int, int, int]:
        def load(key: str, default: int) -> tuple[int, Optional[str]]:
            stored = self.db.get_setting(key)
            if stored is not None:
                try:
                    value = int(stored)
                except ValueError:
                    value = default
            else:
                value = default
            value = max(1, value)
            self.db.set_setting(key, str(value))
            return value, stored

        stitch, stored_stitch = load("stitch_parallelism", DEFAULT_STITCH_CONCURRENCY)
        # migrate legacy max_parallel_jobs if present and no explicit stitch setting
        legacy = self.db.get_setting("max_parallel_jobs")
        if legacy and stored_stitch is None:
            try:
                stitch = max(1, int(legacy))
            except ValueError:
                stitch = stitch
            self.db.set_setting("stitch_parallelism", str(stitch))
        scan, _ = load("scan_parallelism", DEFAULT_SCAN_CONCURRENCY)
        deep_scan, _ = load("deep_scan_parallelism", DEFAULT_DEEP_SCAN_CONCURRENCY)
        thumbnails, _ = load("thumbnail_parallelism", DEFAULT_THUMBNAIL_CONCURRENCY)
        LOGGER.info(
            "Using concurrency stitch=%d scan=%d deep_scan=%d thumbnails=%d",
            stitch,
            scan,
            deep_scan,
            thumbnails,
        )
        return stitch, scan, deep_scan, thumbnails

    def _load_expected_ratio(self) -> float:
        stored = self.db.get_setting("expected_size_ratio")
        value = DEFAULT_EXPECTED_RATIO
        if stored is not None:
            try:
                value = float(stored)
            except ValueError:
                value = DEFAULT_EXPECTED_RATIO
        value = max(value, 0.01)
        self.db.set_setting("expected_size_ratio", f"{value:.6f}")
        LOGGER.info("Using expected_size_ratio=%.4f", value)
        return value

    def set_parallel_jobs(self, count: int) -> None:
        self.set_concurrency(stitch=count)

    def set_concurrency(
        self,
        *,
        stitch: Optional[int] = None,
        scan: Optional[int] = None,
        deep_scan: Optional[int] = None,
        thumbnails: Optional[int] = None,
    ) -> None:
        updates: Dict[str, int] = {}
        with self._lock:
            if stitch is not None:
                if stitch < 1:
                    raise ValueError("stitch_parallelism must be >= 1")
                self.stitch_parallelism = stitch
                updates["stitch_parallelism"] = stitch
            if scan is not None:
                if scan < 1:
                    raise ValueError("scan_parallelism must be >= 1")
                self.scan_parallelism = scan
                updates["scan_parallelism"] = scan
            if deep_scan is not None:
                if deep_scan < 1:
                    raise ValueError("deep_scan_parallelism must be >= 1")
                self.deep_scan_parallelism = deep_scan
                updates["deep_scan_parallelism"] = deep_scan
            if thumbnails is not None:
                if thumbnails < 1:
                    raise ValueError("thumbnail_parallelism must be >= 1")
                self.thumbnail_parallelism = thumbnails
                updates["thumbnail_parallelism"] = thumbnails
        for key, value in updates.items():
            self.db.set_setting(key, str(value))
        if "stitch_parallelism" in updates:
            self._maybe_start_jobs()
        LOGGER.info(
            "Updated concurrency stitch=%d scan=%d deep_scan=%d thumbnails=%d",
            self.stitch_parallelism,
            self.scan_parallelism,
            self.deep_scan_parallelism,
            self.thumbnail_parallelism,
        )

    def set_expected_ratio(self, ratio: float) -> None:
        if ratio <= 0:
            raise ValueError("expected_size_ratio must be > 0")
        with self._lock:
            self.expected_size_ratio = ratio
            self.db.set_setting("expected_size_ratio", f"{ratio:.6f}")
        LOGGER.info("Updated expected_size_ratio to %.4f", ratio)
        threading.Thread(target=self.recalculate_expected_sizes, daemon=True).start()

    def _load_stitch_settings(self) -> tuple[str, str, str, bool, bool]:
        output_size = self.db.get_setting("output_size") or DEFAULT_OUTPUT_SIZE
        bitrate = self.db.get_setting("bitrate") or DEFAULT_BITRATE
        stitch_type = self.db.get_setting("stitch_type") or DEFAULT_STITCH_TYPE
        auto_raw = self.db.get_setting("auto_resolution")
        auto_resolution = (
            auto_raw.lower() in {"1", "true", "yes", "on"} if auto_raw else DEFAULT_AUTO_RESOLUTION
        )
        bitrate_mode_raw = self.db.get_setting("original_bitrate")
        original_bitrate = (
            bitrate_mode_raw.lower() in {"1", "true", "yes", "on"}
            if bitrate_mode_raw
            else DEFAULT_ORIGINAL_BITRATE
        )
        self.db.set_setting("output_size", output_size)
        self.db.set_setting("bitrate", bitrate)
        self.db.set_setting("stitch_type", stitch_type)
        self.db.set_setting("auto_resolution", "1" if auto_resolution else "0")
        self.db.set_setting("original_bitrate", "1" if original_bitrate else "0")
        LOGGER.info(
            "Using stitch settings output_size=%s bitrate=%s stitch_type=%s auto=%s original_bitrate=%s",
            output_size,
            bitrate,
            stitch_type,
            auto_resolution,
            original_bitrate,
        )
        return output_size, bitrate, stitch_type, auto_resolution, original_bitrate

    def set_stitch_settings(
        self,
        output_size: Optional[str],
        bitrate: str,
        stitch_type: str,
        auto_resolution: bool,
        original_bitrate: bool,
    ) -> None:
        output_size = (output_size or "").strip()
        bitrate = (bitrate or "").strip()
        stitch_type = (stitch_type or "").strip()
        if not stitch_type:
            raise ValueError("stitch_type must be provided")
        if not original_bitrate and not bitrate:
            raise ValueError("bitrate must be provided when original bitrate is disabled")
        if not auto_resolution and not output_size:
            raise ValueError("output_size must be provided when auto resolution is disabled")
        if not output_size:
            output_size = self.output_size or DEFAULT_OUTPUT_SIZE
        with self._lock:
            self.output_size = output_size
            self.bitrate = bitrate
            self.stitch_type = stitch_type
            self.auto_resolution = bool(auto_resolution)
            self.original_bitrate = bool(original_bitrate)
            self.db.set_setting("output_size", output_size)
            self.db.set_setting("bitrate", bitrate)
            self.db.set_setting("stitch_type", stitch_type)
            self.db.set_setting("auto_resolution", "1" if self.auto_resolution else "0")
            self.db.set_setting("original_bitrate", "1" if self.original_bitrate else "0")
        LOGGER.info(
            "Updated stitch settings output_size=%s bitrate=%s stitch_type=%s auto=%s original_bitrate=%s",
            output_size,
            bitrate,
            stitch_type,
            self.auto_resolution,
            self.original_bitrate,
        )

    def _base_input_size(self, sources: Iterable[str]) -> Optional[int]:
        paths = list(sources)
        if not paths:
            return 0
        total = 0
        for path in paths:
            try:
                total += os.path.getsize(path)
            except FileNotFoundError:
                LOGGER.debug("Missing source %s while computing base size", path)
                return None
        # For single-container files we assume the two camera streams share the bytes equally,
        # so the combined size is still the file size.
        return total

    def _calculate_expected_size(self, sources: Iterable[str], base_size: Optional[int] = None) -> int:
        size = base_size
        if size is None:
            size = self._base_input_size(sources)
        if size is None:
            return 0
        return int(round(size * self.expected_size_ratio))

    def recalculate_expected_sizes(self, job_ids: Optional[Iterable[str]] = None) -> None:
        rows = (
            self.db.fetch_jobs_by_ids(job_ids)
            if job_ids is not None
            else self.db.fetch_jobs()
        )
        for row in rows:
            sources = json.loads(row["source_files"])
            base_size = self._base_input_size(sources)
            expected_size = self._calculate_expected_size(sources, base_size)
            stitched_size = row["stitched_size"] or 0
            final_file = row["final_file"]
            if stitched_size == 0 and final_file and os.path.exists(final_file):
                stitched_size = os.path.getsize(final_file)
            process_ratio = min(stitched_size / expected_size, 1.0) if expected_size else 0
            current_expected = row["expected_size"] or 0
            current_process = row["process"] or 0
            updates: Dict[str, object] = {}
            if current_expected != expected_size:
                updates["expected_size"] = expected_size
            if abs(current_process - process_ratio) >= 1e-4:
                updates["process"] = process_ratio
            if stitched_size and stitched_size != (row["stitched_size"] or 0):
                updates["stitched_size"] = stitched_size
            if not updates:
                continue
            self.db.update_job(row["id"], **updates)
            LOGGER.debug(
                "Job %s expected size refreshed to %d (progress=%.3f)",
                row["id"],
                expected_size,
                process_ratio,
            )

    def compute_expected_ratio(self) -> float:
        rows = self.db.fetch_jobs(statuses=[STATUS_PROCESSED])
        ratios: List[float] = []
        for row in rows:
            sources = json.loads(row["source_files"])
            base_size = self._base_input_size(sources)
            if not base_size:
                continue
            final_file = row["final_file"]
            stitched_size = (
                os.path.getsize(final_file)
                if final_file and os.path.exists(final_file)
                else row["stitched_size"] or 0
            )
            if not stitched_size:
                continue
            ratios.append(stitched_size / base_size)
        count = len(ratios)
        if not count:
            raise ValueError("No completed jobs available to compute ratio")
        avg_ratio = sum(ratios) / count
        variance = sum((value - avg_ratio) ** 2 for value in ratios) / count
        std_dev = variance ** 0.5
        self.set_expected_ratio(avg_ratio)
        LOGGER.info(
            "Computed expected_size_ratio=%.4f (std=%.4f) from %d completed jobs",
            avg_ratio,
            std_dev,
            count,
        )
        return avg_ratio, std_dev

    def _auto_output_size(self, sources: Iterable[str]) -> Optional[str]:
        for path in sources:
            resolution = video_resolution(path)
            if not resolution:
                continue
            width, height = resolution
            if width <= 0 or height <= 0:
                continue
            derived_width = max(2 * width, 2)
            derived_height = max(height, 2)
            output = f"{derived_width}x{derived_height}"
            LOGGER.debug("Derived auto output size %s from %s", output, path)
            return output
        LOGGER.warning("Unable to derive auto output size from sources %s", list(sources))
        return None

    # ------------------------------ Discovery ------------------------------
    def discover_candidates(self, existing_final_files: Optional[Set[str]] = None) -> List[JobCandidate]:
        candidates: Dict[str, JobCandidate] = {}
        existing = existing_final_files or set()
        try:
            raw_files = os.listdir(RAW_DIR)
        except FileNotFoundError:
            LOGGER.warning("RAW directory %s does not exist; skipping scan", RAW_DIR)
            return []

        LOGGER.debug("Scanning %s for candidate pairs", RAW_DIR)
        for file_name in raw_files:
            match = FILE_PATTERN.match(file_name)
            if not match:
                continue
            timestamp, segment = match.groups()
            final_file = stitched_path(timestamp)
            if final_file in existing:
                LOGGER.debug("Candidate %s already tracked; skipping early", final_file)
                continue
            camera_00 = os.path.join(RAW_DIR, file_name)
            stream_count = count_video_streams(camera_00)
            if stream_count is None:
                LOGGER.debug("Unable to determine stream count for %s; skipping", camera_00)
                continue
            if stream_count >= 2:
                source_files = [camera_00]
                LOGGER.debug(
                    "Detected %d video streams in %s; single-file candidate",
                    stream_count,
                    camera_00,
                )
            elif stream_count == 1:
                camera_10 = os.path.join(RAW_DIR, f"VID_{timestamp}_10_{segment}.insv")
                if not os.path.exists(camera_10):
                    LOGGER.debug(
                        "Single-stream file %s missing counterpart %s; skipping",
                        camera_00,
                        camera_10,
                    )
                    continue
                source_files = [camera_00, camera_10]
            else:
                LOGGER.warning(
                    "File %s reported %d video streams; skipping", camera_00, stream_count
                )
                continue
            key = f"{timestamp}_{segment}"
            candidates[key] = JobCandidate(timestamp, segment, source_files)
            LOGGER.debug(
                "Found candidate %s segment %s with %d source(s)",
                timestamp,
                segment,
                len(source_files),
            )
        return list(candidates.values())

    def scan(self, cancel_event: Optional[threading.Event] = None) -> Dict[str, int]:
        if cancel_event and cancel_event.is_set():
            LOGGER.info("Scan cancelled before start")
            return {"added": 0}
        added = 0
        LOGGER.info("Running scan for new RAW pairs")
        existing = self.db.list_final_files()
        candidates = self.discover_candidates(existing)
        if not candidates:
            LOGGER.info("No new candidates discovered")
            return {"added": 0}

        def process_candidate(candidate: JobCandidate) -> int:
            if cancel_event and cancel_event.is_set():
                return 0
            final_file = candidate.final_file
            if self.db.find_by_final_file(final_file):
                LOGGER.debug("Job for %s already exists; skipping", final_file)
                return 0
            if any(is_file_writing(path) for path in candidate.sources()):
                LOGGER.debug("Candidate %s still writing; delaying", candidate.timestamp)
                return 0
            base_size = self._base_input_size(candidate.sources())
            if base_size is None:
                LOGGER.warning("Candidate %s missing source during scan; skipping", candidate.timestamp)
                return 0
            expected_size = self._calculate_expected_size(candidate.sources(), base_size)
            self.db.insert_job(
                timestamp=candidate.timestamp,
                final_file=final_file,
                source_files=candidate.sources(),
                status=STATUS_UNPROCESSED,
                expected_size=expected_size,
            )
            LOGGER.info("Queued job %s with expected size %d", candidate.timestamp, expected_size)
            return 1

        futures = []
        with ThreadPoolExecutor(max_workers=self.scan_parallelism) as executor:
            for candidate in candidates:
                futures.append(executor.submit(process_candidate, candidate))
            for future in as_completed(futures):
                if cancel_event and cancel_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    LOGGER.info("Scan cancelled; stopping remaining candidates")
                    break
                added += future.result()
        LOGGER.info("Scan complete; %d new jobs added", added)
        return {"added": added}

    def deep_scan(self, cancel_event: Optional[threading.Event] = None) -> Dict[str, int]:
        if cancel_event and cancel_event.is_set():
            LOGGER.info("Deep scan cancelled before start")
            return {"added": 0, "updated": 0}
        LOGGER.info("Running deep scan to refresh existing jobs")
        summary = self.scan(cancel_event)
        if cancel_event and cancel_event.is_set():
            summary["updated"] = 0
            return summary
        rows = self.db.fetch_jobs()

        def refresh_row(row: sqlite3.Row) -> int:
            if cancel_event and cancel_event.is_set():
                return 0
            sources = json.loads(row["source_files"])
            missing_sources = [path for path in sources if not os.path.exists(path)]
            base_size = None if missing_sources else self._base_input_size(sources)
            expected_size = self._calculate_expected_size(sources, base_size)
            output_size = os.path.getsize(row["final_file"]) if os.path.exists(row["final_file"]) else 0
            ratio = min(output_size / expected_size, 1.0) if expected_size else 0
            new_status = row["status"]
            if missing_sources:
                new_status = STATUS_FAILED
            elif output_size:
                if ratio >= MIN_SUCCESS_RATIO:
                    new_status = STATUS_PROCESSED
                else:
                    new_status = STATUS_UNPROCESSED
            else:
                new_status = STATUS_UNPROCESSED
            fields: Dict[str, object] = {
                "expected_size": expected_size,
                "stitched_size": output_size,
                "process": ratio,
                "status": new_status,
            }
            if missing_sources:
                fields["pid"] = None
            self.db.update_job(row["id"], **fields)
            LOGGER.debug(
                "Job %s status updated to %s (ratio=%.2f)", row["id"], new_status, ratio
            )
            return 1

        updated = 0
        futures = []
        with ThreadPoolExecutor(max_workers=self.deep_scan_parallelism) as executor:
            for row in rows:
                futures.append(executor.submit(refresh_row, row))
            for future in as_completed(futures):
                if cancel_event and cancel_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    LOGGER.info("Deep scan cancelled; stopping remaining updates")
                    break
                updated += future.result()
        summary["updated"] = updated
        LOGGER.info("Deep scan refreshed %d jobs", updated)
        return summary

    # ------------------------------ Stitching -----------------------------
    def stitch(self, include_failed: bool = False, cancel_event: Optional[threading.Event] = None) -> Dict[str, int]:
        statuses = [STATUS_UNPROCESSED]
        if include_failed:
            statuses.append(STATUS_FAILED)
        jobs = self.db.fetch_jobs(statuses=statuses)
        LOGGER.info("Scheduling stitch run for %d jobs (include_failed=%s)", len(jobs), include_failed)
        queued = 0
        for job in jobs:
            if cancel_event and cancel_event.is_set():
                LOGGER.info("Stitch scheduling cancelled after %d queued jobs", queued)
                break
            queued += int(self._enqueue_job(job["id"]))
        if queued:
            LOGGER.info("Queued %d jobs for stitching", queued)
        self._maybe_start_jobs()
        return {"scheduled": queued}

    def _enqueue_job(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._active_threads or job_id in self._pending_job_ids:
                return False
            self._pending_jobs.append(job_id)
            self._pending_job_ids.add(job_id)
        LOGGER.debug("Job %s enqueued for stitching", job_id)
        return True

    def _maybe_start_jobs(self) -> None:
        while True:
            with self._lock:
                if len(self._active_threads) >= self.stitch_parallelism or not self._pending_jobs:
                    return
                job_id = self._pending_jobs.popleft()
                self._pending_job_ids.discard(job_id)
                if job_id in self._active_threads:
                    continue
                worker = threading.Thread(target=self._run_job, args=(job_id,), daemon=False)
                self._active_threads[job_id] = worker
            worker.start()
            LOGGER.info("Started stitching thread %s for job %s", worker.name, job_id)

    def _run_job(self, job_id: str) -> None:
        try:
            row = self.db.fetch_job(job_id)
            if not row:
                LOGGER.error("Unable to load job %s", job_id)
                return
            sources = json.loads(row["source_files"])
            if not sources:
                self.db.update_job(job_id, status=STATUS_FAILED)
                LOGGER.error("Job %s has no source files listed", job_id)
                return
            if len(sources) > 2:
                self.db.update_job(job_id, status=STATUS_FAILED)
                LOGGER.error("Job %s has unsupported source list: %s", job_id, sources)
                return
            if not all(os.path.exists(path) for path in sources):
                self.db.update_job(job_id, status=STATUS_FAILED, pid=None)
                LOGGER.warning("Job %s missing source files; marking failed", job_id)
                return
            if any(is_file_writing(path) for path in sources):
                time.sleep(5)
            if len(sources) == 2:
                file1, file2 = sources
                if abs(os.path.getsize(file1) - os.path.getsize(file2)) > 0.2 * os.path.getsize(file1):
                    self.db.update_job(job_id, status=STATUS_FAILED)
                    LOGGER.warning("Job %s source files have mismatched sizes", job_id)
                    return
            final_file = row["final_file"]
            temp_output = f"{final_file}.{uuid.uuid4()}.tmp"
            log_file = log_path(row["timestamp"])
            LOGGER.info("Job %s starting stitch to %s", job_id, final_file)
            output_size_value = self.output_size or DEFAULT_OUTPUT_SIZE
            if self.auto_resolution:
                derived = self._auto_output_size(sources)
                if derived:
                    output_size_value = derived
                else:
                    LOGGER.warning(
                        "Falling back to manual output size %s for job %s", output_size_value, job_id
                    )
            cmd = [
                "MediaSDKTest",
                "-inputs",
                *sources,
                "-output_size",
                output_size_value,
            ]
            if not self.original_bitrate and self.bitrate:
                cmd.extend(["-bitrate", self.bitrate])
            cmd.extend(
                [
                    "-stitch_type",
                    self.stitch_type,
                    "-output",
                    temp_output,
                ]
            )
            if self.debug_mode:
                LOGGER.info("Debug mode enabled; suppressing MediaSDKTest execution for job %s", job_id)
                LOGGER.debug("Suppressed command: %s", " ".join(cmd))
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, "w") as log_handle:
                    log_handle.write("Debug mode: MediaSDKTest invocation suppressed.\n")
                    log_handle.write("Command: " + " ".join(cmd) + "\n")
                expected_size = self._expected_size(job_id)
                ratio = 1.0 if expected_size else 0.0
                self.db.update_job(
                    job_id,
                    status=STATUS_PROCESSED,
                    pid=None,
                    stitched_size=expected_size,
                    process=ratio,
                )
                LOGGER.info("Job %s marked processed in debug mode (ratio=%.2f)", job_id, ratio)
                return
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "w") as log_handle:
                process = subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle)
            self._job_temp_outputs[job_id] = temp_output
            self.db.update_job(job_id, status=STATUS_PROCESSING, pid=process.pid)
            LOGGER.debug("Job %s running process %d", job_id, process.pid)
            try:
                while True:
                    ret = process.poll()
                    self._refresh_progress(job_id)
                    if ret is not None:
                        break
                    time.sleep(5)
            finally:
                process.wait()
                self._refresh_progress(job_id)
                output_size = os.path.getsize(temp_output) if os.path.exists(temp_output) else 0
                expected_size = self._expected_size(job_id)
                ratio = min(output_size / expected_size, 1.0) if expected_size else 0
                status = STATUS_FAILED
                if process.returncode == 0 and os.path.exists(temp_output):
                    timestamp_iso = format_creation_time(row["timestamp"])
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_output,
                        "-c",
                        "copy",
                        "-map",
                        "0",
                        "-metadata",
                        f"creation_time={timestamp_iso}",
                        "-metadata",
                        "spherical=true",
                        "-metadata",
                        "stitched=true",
                        "-metadata",
                        "projection_type=equirectangular",
                        "-metadata",
                        "stereo_mode=mono",
                        "-metadata",
                        "stitching_software=Insta360-AutoStitcher",
                        "-metadata:s:v:0",
                        f"creation_time={timestamp_iso}",
                        "-metadata:s:v:0",
                        "spherical=true",
                        "-metadata:s:v:0",
                        "stitched=true",
                        "-metadata:s:v:0",
                        "projection_type=equirectangular",
                        "-metadata:s:v:0",
                        "stereo_mode=mono",
                        "-metadata:s:v:0",
                        "stitching_software=Insta360-AutoStitcher",
                        "-metadata:s:a:0",
                        f"creation_time={timestamp_iso}",
                        final_file,
                    ]
                    ffmpeg_proc = subprocess.run(ffmpeg_cmd, capture_output=True)  # noqa: S603
                    if ffmpeg_proc.returncode != 0:
                        LOGGER.error(
                            "ffmpeg metadata injection failed for job %s: %s",
                            job_id,
                            ffmpeg_proc.stderr.decode(errors="ignore"),
                        )
                        status = STATUS_FAILED
                    else:
                        inject_spherical_metadata(final_file)
                        status = STATUS_PROCESSED
                if status == STATUS_FAILED and os.path.exists(temp_output):
                    try:
                        os.remove(temp_output)
                    except OSError:
                        LOGGER.warning("Failed to remove temp output %s for job %s", temp_output, job_id)
                elif status == STATUS_PROCESSED and os.path.exists(temp_output):
                    try:
                        os.remove(temp_output)
                    except OSError:
                        LOGGER.warning("Failed to remove temp output %s after success for job %s", temp_output, job_id)
                self.db.update_job(
                    job_id,
                    status=status,
                    pid=None,
                    stitched_size=os.path.getsize(final_file) if os.path.exists(final_file) else output_size,
                    process=ratio,
                )
                LOGGER.info(
                    "Job %s finished with status %s (ratio=%.2f, return=%s)",
                    job_id,
                    status,
                    ratio,
                    process.returncode,
                )
        finally:
            with self._lock:
                self._active_threads.pop(job_id, None)
            self._job_temp_outputs.pop(job_id, None)
            self._maybe_start_jobs()

    def _refresh_progress(self, job_id: str) -> None:
        row = self.db.fetch_job(job_id)
        if not row:
            return
        with self._lock:
            temp_path = self._job_temp_outputs.get(job_id)
        candidate = temp_path if temp_path and os.path.exists(temp_path) else row["final_file"]
        if not os.path.exists(candidate):
            return
        output_size = os.path.getsize(candidate)
        expected = self._expected_size(job_id)
        ratio = min(output_size / expected, 1.0) if expected else 0
        self.db.update_job(
            job_id,
            stitched_size=output_size,
            process=ratio,
        )
        LOGGER.debug("Job %s progress updated: %d/%d bytes (%.2f)", job_id, output_size, expected, ratio)

    def _expected_size(self, job_id: str) -> int:
        row = self.db.fetch_job(job_id)
        if not row:
            return 0
        sources = json.loads(row["source_files"])
        base_size = self._base_input_size(sources)
        expected = self._calculate_expected_size(sources, base_size)
        current_expected = row["expected_size"] or 0
        if expected != current_expected:
            self.db.update_job(job_id, expected_size=expected)
            LOGGER.debug("Job %s expected size recalculated to %d", job_id, expected)
        return expected

    # ------------------------------- REST ---------------------------------
    def serialize_job(self, row: sqlite3.Row) -> Dict[str, object]:
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "final_file": row["final_file"],
            "source_files": json.loads(row["source_files"]),
            "status": row["status"],
            "pid": row["pid"],
            "stitched_size": row["stitched_size"],
            "process": row["process"],
            "expected_size": row["expected_size"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "thumbnail_url": f"/thumbnails/{row['id']}.jpg" if os.path.exists(thumbnail_path(row["id"])) else None,
        }

    def start_task(self, action: str, job_ids: Optional[List[str]] = None) -> str:
        task_id = str(uuid.uuid4())
        cancel_event = threading.Event()
        started_at = utc_now()

        def runner() -> None:
            try:
                self.run_action(action, job_ids, cancel_event)
            finally:
                with self._task_lock:
                    self._active_tasks.pop(task_id, None)

        thread = threading.Thread(target=runner, daemon=True)
        with self._task_lock:
            self._active_tasks[task_id] = ActiveTask(
                task_id=task_id,
                action=action,
                started_at=started_at,
                cancel_event=cancel_event,
                thread=thread,
            )
        thread.start()
        return task_id

    def cancel_task(self, task_id: str) -> bool:
        with self._task_lock:
            task = self._active_tasks.get(task_id)
        if not task:
            return False
        task.cancel_event.set()
        return True

    def list_active_tasks(self) -> List[Dict[str, str]]:
        with self._task_lock:
            tasks = list(self._active_tasks.values())
        return [
            {"id": task.task_id, "action": task.action, "started_at": task.started_at}
            for task in tasks
        ]

    def get_status(self) -> Dict[str, object]:
        jobs = [self.serialize_job(row) for row in self.db.fetch_jobs()]
        with self._lock:
            active_ids = set(self._active_threads.keys())
            queued_ids = set(self._pending_job_ids)
        eligible_statuses = {STATUS_UNPROCESSED, STATUS_FAILED}
        queued_jobs = len(queued_ids)
        waiting_jobs = 0
        for job in jobs:
            if job["id"] in queued_ids:
                job["queue_state"] = "queued"
            elif job["status"] in eligible_statuses and job["id"] not in active_ids:
                job["queue_state"] = "pending"
                waiting_jobs += 1
            else:
                job["queue_state"] = None
        return {
            "jobs": jobs,
            "active_jobs": list(active_ids),
            "pending_jobs": waiting_jobs,
            "queued_jobs": queued_jobs,
            "max_parallel_jobs": self.stitch_parallelism,
            "expected_size_ratio": self.expected_size_ratio,
            "active_tasks": self.list_active_tasks(),
            "stitch_settings": {
                "output_size": self.output_size,
                "bitrate": self.bitrate,
                "stitch_type": self.stitch_type,
                "auto_resolution": self.auto_resolution,
                "original_bitrate": self.original_bitrate,
            },
            "concurrency": {
                "stitch": self.stitch_parallelism,
                "scan": self.scan_parallelism,
                "deep_scan": self.deep_scan_parallelism,
                "thumbnails": self.thumbnail_parallelism,
            },
        }

    def run_action(
        self,
        action: str,
        job_ids: Optional[List[str]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, int]:
        LOGGER.info("Running action %s", action)
        if action == "scan":
            return self.scan(cancel_event)
        if action == "deep_scan":
            return self.deep_scan(cancel_event)
        if action == "stitch":
            return self.stitch(include_failed=False, cancel_event=cancel_event)
        if action == "full_stitch":
            return self.stitch(include_failed=True, cancel_event=cancel_event)
        if action == "generate_thumbnails":
            return self.generate_thumbnails(job_ids, cancel_event=cancel_event)
        if action == "stitch_selected":
            return self.stitch_selected(job_ids or [], cancel_event=cancel_event)
        raise ValueError(f"Unknown action: {action}")

    def generate_thumbnails(
        self,
        job_ids: Optional[List[str]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, int]:
        if cancel_event and cancel_event.is_set():
            LOGGER.info("Thumbnail generation cancelled before start")
            return {"generated": 0, "skipped": 0, "failed": 0}
        os.makedirs(THUMBNAIL_DIR, exist_ok=True)
        rows = self.db.fetch_jobs_by_ids(job_ids) if job_ids else self.db.fetch_jobs()
        generated = 0
        skipped = 0
        failed = 0

        def process_thumbnail(row: sqlite3.Row) -> tuple[int, int, int]:
            if cancel_event and cancel_event.is_set():
                return (0, 0, 0)
            thumb = thumbnail_path(row["id"])
            if os.path.exists(thumb):
                return (0, 1, 0)
            sources = json.loads(row["source_files"])
            if not sources:
                LOGGER.debug("Job %s has no sources; skipping thumbnail", row["id"])
                return (0, 0, 1)
            primary = sources[0]
            if not os.path.exists(primary):
                LOGGER.debug("Primary source %s missing for job %s", primary, row["id"])
                return (0, 0, 1)
            duration = video_duration_seconds(primary) or 0
            timestamp = duration / 2 if duration > 0 else 0
            if extract_thumbnail(primary, thumb, timestamp):
                LOGGER.info("Generated thumbnail for job %s at %s", row["id"], thumb)
                return (1, 0, 0)
            return (0, 0, 1)

        futures = []
        with ThreadPoolExecutor(max_workers=self.thumbnail_parallelism) as executor:
            for row in rows:
                futures.append(executor.submit(process_thumbnail, row))
            for future in as_completed(futures):
                if cancel_event and cancel_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    LOGGER.info("Thumbnail generation cancelled; stopping remaining jobs")
                    break
                gen, skip, fail = future.result()
                generated += gen
                skipped += skip
                failed += fail
        if job_ids:
            LOGGER.info(
                "Thumbnail generation complete for selected jobs (generated=%d, skipped=%d, failed=%d)",
                generated,
                skipped,
                failed,
            )
        else:
            LOGGER.info(
                "Thumbnail generation complete (generated=%d, skipped=%d, failed=%d)",
                generated,
                skipped,
                failed,
            )
        return {"generated": generated, "skipped": skipped, "failed": failed}

    def stitch_selected(
        self,
        job_ids: List[str],
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, int]:
        if not job_ids:
            LOGGER.info("No job IDs provided for stitch_selected")
            return {"scheduled": 0}
        rows = self.db.fetch_jobs_by_ids(job_ids)
        queued = 0
        allowed_statuses = {STATUS_UNPROCESSED, STATUS_FAILED}
        for row in rows:
            if cancel_event and cancel_event.is_set():
                LOGGER.info("Stitch selected cancelled after %d queued jobs", queued)
                break
            if row["status"] not in allowed_statuses:
                LOGGER.debug(
                    "Skipping job %s for stitch_selected due to status %s",
                    row["id"],
                    row["status"],
                )
                continue
            queued += int(self._enqueue_job(row["id"]))
        if queued:
            LOGGER.info("Queued %d selected jobs for stitching", queued)
        self._maybe_start_jobs()
        return {"scheduled": queued}


def create_app(controller: AutoStitcher) -> Flask:
    app = Flask(__name__)

    @app.before_request
    def log_request() -> None:
        if LOGIN_TOKEN and request.path != "/login":
            auth_header = request.headers.get("Authorization", "")
            token = ""
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1].strip()
            if not token:
                token = request.headers.get("X-Auth-Token", "").strip()
            if token != LOGIN_TOKEN:
                LOGGER.warning("Unauthorized request to %s", request.path)
                return jsonify({"error": "unauthorized"}), 401
        LOGGER.debug(">> %s %s", request.method, request.path)

    @app.after_request
    def log_response(response):
        LOGGER.debug("<< %s %s %s", request.method, request.path, response.status)
        return response

    @app.get("/status")
    def status() -> object:
        LOGGER.debug("Status endpoint queried")
        return jsonify(controller.get_status())

    @app.post("/tasks")
    def tasks() -> object:
        payload = request.get_json(silent=True) or {}
        action = payload.get("action")
        if action not in {"scan", "deep_scan", "stitch", "full_stitch", "generate_thumbnails", "stitch_selected"}:
            LOGGER.warning("Received invalid task action: %s", action)
            return jsonify({"error": "unknown action"}), 400
        job_ids = payload.get("job_ids")
        if action == "stitch_selected":
            if not isinstance(job_ids, list):
                LOGGER.warning("Invalid job_ids payload for stitch_selected: %s", job_ids)
                return jsonify({"error": "job_ids must be a list"}), 400
            job_ids = [str(jid) for jid in job_ids]
        elif action == "generate_thumbnails":
            if job_ids is not None and not isinstance(job_ids, list):
                LOGGER.warning("Invalid job_ids payload for generate_thumbnails: %s", job_ids)
                return jsonify({"error": "job_ids must be a list"}), 400
            if isinstance(job_ids, list):
                job_ids = [str(jid) for jid in job_ids]
            else:
                job_ids = None
        else:
            job_ids = None
        LOGGER.info("Scheduling action %s from REST request", action)
        task_id = controller.start_task(action, job_ids)
        return jsonify({"scheduled": action, "task_id": task_id})

    @app.post("/tasks/terminate")
    def terminate_task() -> object:
        payload = request.get_json(silent=True) or {}
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            LOGGER.warning("Invalid task_id for termination: %s", task_id)
            return jsonify({"error": "task_id must be provided"}), 400
        if controller.cancel_task(task_id):
            LOGGER.info("Termination requested for task %s", task_id)
            return jsonify({"terminated": task_id})
        LOGGER.warning("Termination requested for unknown task %s", task_id)
        return jsonify({"error": "unknown task"}), 404

    @app.post("/login")
    def login() -> object:
        if not LOGIN_TOKEN:
            return jsonify({"ok": True})
        payload = request.get_json(silent=True) or {}
        token = payload.get("token")
        if token != LOGIN_TOKEN:
            return jsonify({"error": "invalid token"}), 401
        return jsonify({"ok": True})

    @app.get("/thumbnails/<job_id>.jpg")
    def thumbnails(job_id: str):
        path = thumbnail_path(job_id)
        if not os.path.exists(path):
            LOGGER.debug("Thumbnail for job %s not found", job_id)
            return ("", 404)
        return send_from_directory(THUMBNAIL_DIR, f"{job_id}.jpg")

    @app.get("/settings/parallelism")
    def get_parallelism() -> object:
        return jsonify(
            {
                "stitch_parallelism": controller.stitch_parallelism,
                "scan_parallelism": controller.scan_parallelism,
                "deep_scan_parallelism": controller.deep_scan_parallelism,
                "thumbnail_parallelism": controller.thumbnail_parallelism,
                "max_parallel_jobs": controller.stitch_parallelism,
            }
        )

    @app.post("/settings/parallelism")
    def set_parallelism() -> object:
        payload = request.get_json(silent=True) or {}
        updates: Dict[str, int] = {}
        mappings = {
            "stitch_parallelism": "stitch",
            "scan_parallelism": "scan",
            "deep_scan_parallelism": "deep_scan",
            "thumbnail_parallelism": "thumbnails",
        }
        for payload_key, field in mappings.items():
            if payload_key in payload:
                value = payload[payload_key]
                try:
                    updates[field] = int(value)
                except (TypeError, ValueError):
                    LOGGER.warning("Invalid %s payload: %s", payload_key, payload)
                    return jsonify({ "error": f"{payload_key} must be an integer"}), 400
        if "max_parallel_jobs" in payload and "stitch" not in updates:
            try:
                updates["stitch"] = int(payload["max_parallel_jobs"])
            except (TypeError, ValueError):
                LOGGER.warning("Invalid max_parallel_jobs payload: %s", payload)
                return jsonify({"error": "max_parallel_jobs must be an integer"}), 400
        if not updates:
            return jsonify({"error": "No parallelism fields provided"}), 400
        try:
            controller.set_concurrency(**updates)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            {
                "stitch_parallelism": controller.stitch_parallelism,
                "scan_parallelism": controller.scan_parallelism,
                "deep_scan_parallelism": controller.deep_scan_parallelism,
                "thumbnail_parallelism": controller.thumbnail_parallelism,
                "max_parallel_jobs": controller.stitch_parallelism,
            }
        )

    @app.get("/settings/ratio")
    def get_ratio() -> object:
        return jsonify({"expected_size_ratio": controller.expected_size_ratio})

    @app.post("/settings/ratio")
    def set_ratio() -> object:
        payload = request.get_json(silent=True) or {}
        value = payload.get("expected_size_ratio")
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid ratio payload: %s", payload)
            return jsonify({"error": "expected_size_ratio must be a number"}), 400
        try:
            controller.set_expected_ratio(ratio)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"expected_size_ratio": controller.expected_size_ratio})

    @app.post("/settings/ratio/compute")
    def compute_ratio() -> object:
        try:
            ratio, std_dev = controller.compute_expected_ratio()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"expected_size_ratio": ratio, "std_dev": std_dev})

    @app.get("/settings/stitch")
    def get_stitch_settings() -> object:
        return jsonify(
            {
                "output_size": controller.output_size,
                "bitrate": controller.bitrate,
                "stitch_type": controller.stitch_type,
            }
        )

    @app.post("/settings/stitch")
    def set_stitch_settings() -> object:
        payload = request.get_json(silent=True) or {}
        output_size = payload.get("output_size")
        bitrate = payload.get("bitrate")
        stitch_type = payload.get("stitch_type")
        auto_resolution = payload.get("auto_resolution", False)
        original_bitrate = payload.get("original_bitrate", False)
        try:
            controller.set_stitch_settings(
                output_size, bitrate, stitch_type, bool(auto_resolution), bool(original_bitrate)
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            {
                "output_size": controller.output_size,
                "bitrate": controller.bitrate,
                "stitch_type": controller.stitch_type,
                "auto_resolution": controller.auto_resolution,
                "original_bitrate": controller.original_bitrate,
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Insta360 autostitcher controller")
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR). Defaults to AUTO_STITCHER_LOG_LEVEL env var.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (skip MediaSDKTest execution, force DEBUG logging). "
        "Can also be enabled via AUTO_STITCHER_DEBUG env var.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Directory containing RAW INSV files. Defaults to RAW_DIR env var ",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for stitched MP4s and logs. Defaults to OUT_DIR env var ",
    )
    parser.add_argument(
        "--storage-dir",
        default=None,
        help=(
            "Directory that will contain the database, RAW, and stitched folders. "
            "Defaults to APP_STORAGE_DIR env var or /app."
        ),
    )
    parser.add_argument(
        "command",
        choices=["serve", "scan", "deep_scan", "stitch", "full_stitch", "generate_thumbnails", "stitch_selected"],
        nargs="?",
        default="serve",
        help="Run as REST server (default) or execute a one-off command",
    )
    args = parser.parse_args()
    debug_mode = args.debug or DEBUG_FLAG
    log_level = "DEBUG" if debug_mode else args.log_level
    configure_logging(log_level)
    LOGGER.info("Logger configured at %s level (debug=%s)", log_level.upper(), debug_mode)
    configure_paths(
        storage_dir=args.storage_dir,
        raw_dir=args.raw_dir,
        out_dir=args.output_dir,
    )
    LOGGER.info(
        "Using storage directory %s (RAW=%s, OUT=%s, DB=%s)",
        APP_STORAGE_DIR,
        RAW_DIR,
        OUT_DIR,
        DATABASE_PATH,
    )
    if debug_mode:
        LOGGER.info("Debug mode active: MediaSDKTest invocations will be skipped.")
    controller = AutoStitcher(debug=debug_mode)
    if args.command == "serve":
        app = create_app(controller)
        app.run(host="0.0.0.0", port=REST_PORT)
        return
    result = controller.run_action(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
