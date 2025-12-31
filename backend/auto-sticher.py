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
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Iterable, List, Optional, Set
from collections import deque

from flask import Flask, jsonify, request, send_from_directory

APP_STORAGE_DIR = os.getenv("APP_STORAGE_DIR", "/app").rstrip("/")
RAW_DIR = os.path.join(APP_STORAGE_DIR, "raw")
OUT_DIR = os.path.join(APP_STORAGE_DIR, "stitched")
DATABASE_PATH = os.getenv(
    "AUTO_STITCHER_DB", os.path.join(APP_STORAGE_DIR, "autostitcher.db")
)
THUMBNAIL_DIR = os.path.join(APP_STORAGE_DIR, "thumbnails")
DEFAULT_PARALLEL_JOBS = max(int(os.getenv("MAX_PARALLEL_JOBS", "1")), 1)
try:
    DEFAULT_EXPECTED_RATIO = float(os.getenv("EXPECTED_SIZE_RATIO", "1.0"))
except ValueError:
    DEFAULT_EXPECTED_RATIO = 1.0
if DEFAULT_EXPECTED_RATIO <= 0:
    DEFAULT_EXPECTED_RATIO = 1.0
DEFAULT_OUTPUT_SIZE = os.getenv("OUTPUT_SIZE", "5760x2880")
DEFAULT_BITRATE = os.getenv("BITRATE", "200000000")
DEFAULT_STITCH_TYPE = os.getenv("STITCH_TYPE", "dynamicstitch")
DEFAULT_AUTO_RESOLUTION = os.getenv("AUTO_RESOLUTION", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "600"))
REST_PORT = int(os.getenv("AUTO_STITCHER_PORT", "8000"))
LOG_LEVEL = os.getenv("AUTO_STITCHER_LOG_LEVEL", "INFO")
DEBUG_FLAG = os.getenv("AUTO_STITCHER_DEBUG", "0").lower() in {"1", "true", "yes", "on"}

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


class JobDatabase:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
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
        with closing(self._connect()) as conn:
            conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", values)
            conn.commit()

    def fetch_jobs(self, statuses: Optional[Iterable[str]] = None) -> List[sqlite3.Row]:
        with closing(self._connect()) as conn:
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                query = f"SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY timestamp"
                rows = conn.execute(query, tuple(statuses)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM jobs ORDER BY timestamp").fetchall()
        return rows

    def fetch_job(self, job_id: str) -> Optional[sqlite3.Row]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row

    def get_setting(self, key: str) -> Optional[str]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
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
        with closing(self._connect()) as conn:
            rows = conn.execute(f"SELECT * FROM jobs WHERE id IN ({placeholders})", job_ids).fetchall()
        return rows

    def find_by_final_file(self, final_file: str) -> Optional[sqlite3.Row]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE final_file = ?", (final_file,)
            ).fetchone()
        return row


class AutoStitcher:
    def __init__(self, debug: bool = False) -> None:
        ensure_dirs()
        self.db = JobDatabase(DATABASE_PATH)
        self._lock = threading.Lock()
        self._active_threads: Dict[str, threading.Thread] = {}
        self._pending_jobs: Deque[str] = deque()
        self._pending_job_ids: Set[str] = set()
        self.debug_mode = debug
        self.max_parallel_jobs = self._load_parallel_jobs()
        self.expected_size_ratio = self._load_expected_ratio()
        (
            self.output_size,
            self.bitrate,
            self.stitch_type,
            self.auto_resolution,
        ) = self._load_stitch_settings()
        LOGGER.info(
            "AutoStitcher initialized with DB at %s (debug=%s)", DATABASE_PATH, self.debug_mode
        )

    def _load_parallel_jobs(self) -> int:
        stored = self.db.get_setting("max_parallel_jobs")
        value = DEFAULT_PARALLEL_JOBS
        if stored is not None:
            try:
                value = int(stored)
            except ValueError:
                value = DEFAULT_PARALLEL_JOBS
        value = max(1, value)
        self.db.set_setting("max_parallel_jobs", str(value))
        LOGGER.info("Using max_parallel_jobs=%d", value)
        return value

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
        if count < 1:
            raise ValueError("max_parallel_jobs must be >= 1")
        with self._lock:
            self.max_parallel_jobs = count
        self.db.set_setting("max_parallel_jobs", str(count))
        LOGGER.info("Updated max_parallel_jobs to %d", count)
        self._maybe_start_jobs()

    def set_expected_ratio(self, ratio: float) -> None:
        if ratio <= 0:
            raise ValueError("expected_size_ratio must be > 0")
        with self._lock:
            self.expected_size_ratio = ratio
            self.db.set_setting("expected_size_ratio", f"{ratio:.6f}")
        LOGGER.info("Updated expected_size_ratio to %.4f", ratio)
        threading.Thread(target=self.recalculate_expected_sizes, daemon=True).start()

    def _load_stitch_settings(self) -> tuple[str, str, str, bool]:
        output_size = self.db.get_setting("output_size") or DEFAULT_OUTPUT_SIZE
        bitrate = self.db.get_setting("bitrate") or DEFAULT_BITRATE
        stitch_type = self.db.get_setting("stitch_type") or DEFAULT_STITCH_TYPE
        auto_raw = self.db.get_setting("auto_resolution")
        auto_resolution = (
            auto_raw.lower() in {"1", "true", "yes", "on"} if auto_raw else DEFAULT_AUTO_RESOLUTION
        )
        self.db.set_setting("output_size", output_size)
        self.db.set_setting("bitrate", bitrate)
        self.db.set_setting("stitch_type", stitch_type)
        self.db.set_setting("auto_resolution", "1" if auto_resolution else "0")
        LOGGER.info(
            "Using stitch settings output_size=%s bitrate=%s stitch_type=%s auto=%s",
            output_size,
            bitrate,
            stitch_type,
            auto_resolution,
        )
        return output_size, bitrate, stitch_type, auto_resolution

    def set_stitch_settings(
        self, output_size: Optional[str], bitrate: str, stitch_type: str, auto_resolution: bool
    ) -> None:
        output_size = (output_size or "").strip()
        bitrate = (bitrate or "").strip()
        stitch_type = (stitch_type or "").strip()
        if not bitrate or not stitch_type:
            raise ValueError("bitrate and stitch_type must be provided")
        if not auto_resolution and not output_size:
            raise ValueError("output_size must be provided when auto resolution is disabled")
        if not output_size:
            output_size = self.output_size or DEFAULT_OUTPUT_SIZE
        with self._lock:
            self.output_size = output_size
            self.bitrate = bitrate
            self.stitch_type = stitch_type
            self.auto_resolution = bool(auto_resolution)
            self.db.set_setting("output_size", output_size)
            self.db.set_setting("bitrate", bitrate)
            self.db.set_setting("stitch_type", stitch_type)
            self.db.set_setting("auto_resolution", "1" if self.auto_resolution else "0")
        LOGGER.info(
            "Updated stitch settings output_size=%s bitrate=%s stitch_type=%s auto=%s",
            output_size,
            bitrate,
            stitch_type,
            self.auto_resolution,
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
    def discover_candidates(self) -> List[JobCandidate]:
        candidates: Dict[str, JobCandidate] = {}
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

    def scan(self) -> Dict[str, int]:
        added = 0
        LOGGER.info("Running scan for new RAW pairs")
        for candidate in self.discover_candidates():
            final_file = candidate.final_file
            if self.db.find_by_final_file(final_file):
                LOGGER.debug("Job for %s already exists; skipping", final_file)
                continue
            if any(is_file_writing(path) for path in candidate.sources()):
                LOGGER.debug("Candidate %s still writing; delaying", candidate.timestamp)
                continue
            base_size = self._base_input_size(candidate.sources())
            if base_size is None:
                LOGGER.warning("Candidate %s missing source during scan; skipping", candidate.timestamp)
                continue
            expected_size = self._calculate_expected_size(candidate.sources(), base_size)
            self.db.insert_job(
                timestamp=candidate.timestamp,
                final_file=final_file,
                source_files=candidate.sources(),
                status=STATUS_UNPROCESSED,
                expected_size=expected_size,
            )
            added += 1
            LOGGER.info("Queued job %s with expected size %d", candidate.timestamp, expected_size)
        LOGGER.info("Scan complete; %d new jobs added", added)
        return {"added": added}

    def deep_scan(self) -> Dict[str, int]:
        LOGGER.info("Running deep scan to refresh existing jobs")
        summary = self.scan()
        updated = 0
        for row in self.db.fetch_jobs():
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
                if ratio >= 1.0:
                    new_status = STATUS_PROCESSED
                elif ratio <= 0.05:
                    new_status = STATUS_FAILED
                else:
                    new_status = STATUS_PROCESSING
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
            updated += 1
            LOGGER.debug(
                "Job %s status updated to %s (ratio=%.2f)", row["id"], new_status, ratio
            )
        summary["updated"] = updated
        LOGGER.info("Deep scan refreshed %d jobs", updated)
        return summary

    # ------------------------------ Stitching -----------------------------
    def stitch(self, include_failed: bool = False) -> Dict[str, int]:
        statuses = [STATUS_UNPROCESSED]
        if include_failed:
            statuses.append(STATUS_FAILED)
        jobs = self.db.fetch_jobs(statuses=statuses)
        LOGGER.info("Scheduling stitch run for %d jobs (include_failed=%s)", len(jobs), include_failed)
        queued = 0
        for job in jobs:
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
                if len(self._active_threads) >= self.max_parallel_jobs or not self._pending_jobs:
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
                "-bitrate",
                self.bitrate,
                "-stitch_type",
                self.stitch_type,
                "-output",
                final_file,
            ]
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
                output_size = os.path.getsize(final_file) if os.path.exists(final_file) else 0
                expected_size = self._expected_size(job_id)
                ratio = min(output_size / expected_size, 1.0) if expected_size else 0
                status = STATUS_PROCESSED if process.returncode == 0 and ratio >= 1.0 else STATUS_FAILED
                self.db.update_job(
                    job_id,
                    status=status,
                    pid=None,
                    stitched_size=output_size,
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
            self._maybe_start_jobs()

    def _refresh_progress(self, job_id: str) -> None:
        row = self.db.fetch_job(job_id)
        if not row:
            return
        final_file = row["final_file"]
        if not os.path.exists(final_file):
            return
        output_size = os.path.getsize(final_file)
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

    def get_status(self) -> Dict[str, object]:
        jobs = [self.serialize_job(row) for row in self.db.fetch_jobs()]
        with self._lock:
            active = list(self._active_threads.keys())
            pending = len(self._pending_jobs)
        return {
            "jobs": jobs,
            "active_jobs": active,
            "pending_jobs": pending,
            "max_parallel_jobs": self.max_parallel_jobs,
            "expected_size_ratio": self.expected_size_ratio,
            "stitch_settings": {
                "output_size": self.output_size,
                "bitrate": self.bitrate,
                "stitch_type": self.stitch_type,
                "auto_resolution": self.auto_resolution,
            },
        }

    def run_action(self, action: str, job_ids: Optional[List[str]] = None) -> Dict[str, int]:
        LOGGER.info("Running action %s", action)
        if action == "scan":
            return self.scan()
        if action == "deep_scan":
            return self.deep_scan()
        if action == "stitch":
            return self.stitch(include_failed=False)
        if action == "full_stitch":
            return self.stitch(include_failed=True)
        if action == "generate_thumbnails":
            return self.generate_thumbnails(job_ids)
        if action == "stitch_selected":
            return self.stitch_selected(job_ids or [])
        raise ValueError(f"Unknown action: {action}")

    def generate_thumbnails(self, job_ids: Optional[List[str]] = None) -> Dict[str, int]:
        os.makedirs(THUMBNAIL_DIR, exist_ok=True)
        rows = self.db.fetch_jobs_by_ids(job_ids) if job_ids else self.db.fetch_jobs()
        generated = 0
        skipped = 0
        failed = 0
        for row in rows:
            thumb = thumbnail_path(row["id"])
            if os.path.exists(thumb):
                skipped += 1
                continue
            sources = json.loads(row["source_files"])
            if not sources:
                LOGGER.debug("Job %s has no sources; skipping thumbnail", row["id"])
                failed += 1
                continue
            primary = sources[0]
            if not os.path.exists(primary):
                LOGGER.debug("Primary source %s missing for job %s", primary, row["id"])
                failed += 1
                continue
            duration = video_duration_seconds(primary) or 0
            timestamp = duration / 2 if duration > 0 else 0
            if extract_thumbnail(primary, thumb, timestamp):
                generated += 1
                LOGGER.info("Generated thumbnail for job %s at %s", row["id"], thumb)
            else:
                failed += 1
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

    def stitch_selected(self, job_ids: List[str]) -> Dict[str, int]:
        if not job_ids:
            LOGGER.info("No job IDs provided for stitch_selected")
            return {"scheduled": 0}
        rows = self.db.fetch_jobs_by_ids(job_ids)
        queued = 0
        allowed_statuses = {STATUS_UNPROCESSED, STATUS_FAILED}
        for row in rows:
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
        threading.Thread(target=controller.run_action, args=(action, job_ids), daemon=True).start()
        return jsonify({"scheduled": action})

    @app.get("/thumbnails/<job_id>.jpg")
    def thumbnails(job_id: str):
        path = thumbnail_path(job_id)
        if not os.path.exists(path):
            LOGGER.debug("Thumbnail for job %s not found", job_id)
            return ("", 404)
        return send_from_directory(THUMBNAIL_DIR, f"{job_id}.jpg")

    @app.get("/settings/parallelism")
    def get_parallelism() -> object:
        return jsonify({"max_parallel_jobs": controller.max_parallel_jobs})

    @app.post("/settings/parallelism")
    def set_parallelism() -> object:
        payload = request.get_json(silent=True) or {}
        value = payload.get("max_parallel_jobs")
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid parallelism payload: %s", payload)
            return jsonify({"error": "max_parallel_jobs must be an integer"}), 400
        try:
            controller.set_parallel_jobs(parsed)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"max_parallel_jobs": controller.max_parallel_jobs})

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
        try:
            controller.set_stitch_settings(output_size, bitrate, stitch_type, bool(auto_resolution))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            {
                "output_size": controller.output_size,
                "bitrate": controller.bitrate,
                "stitch_type": controller.stitch_type,
                "auto_resolution": controller.auto_resolution,
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
