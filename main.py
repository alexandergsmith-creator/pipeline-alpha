#!/usr/bin/env python3
"""
Pipeline Alpha — YouTube Audio Harvest → Kaggle Upload
=======================================================
Pulls audio from YouTube channels, slices into 10-minute chunks,
and versions them into a Kaggle dataset. Designed to run headlessly
on Railway (or any container with yt-dlp, ffmpeg, and Kaggle creds).

Required environment variables (set in Railway → Variables):
  KAGGLE_DATASET   e.g. "username/my-audio-dataset"
  KAGGLE_USERNAME  (also used by the Kaggle SDK automatically)
  KAGGLE_KEY       (also used by the Kaggle SDK automatically)

Optional:
  WORKERS          Parallel download threads   (default: 2)
  FETCH_LIMIT      Videos to pull per channel  (default: 1)
  SEGMENT_SECONDS  Chunk length in seconds     (default: 600)
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from kaggle.api.kaggle_api_extended import KaggleApi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ID: str = os.environ.get("KAGGLE_DATASET", "")
WORKERS: int = int(os.environ.get("WORKERS", "2"))
FETCH_LIMIT: int = int(os.environ.get("FETCH_LIMIT", "1"))
SEGMENT_SECONDS: int = int(os.environ.get("SEGMENT_SECONDS", "600"))

CHANNELS_FILE = Path("channels.json")
ARCHIVE_FILE = Path("archive.txt")

DEFAULT_CHANNELS: list[str] = ["@MKBHD"]

DATASET_METADATA_TITLE = "Pipeline Alpha Harvest"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline-alpha")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PipelineError(RuntimeError):
    """Raised when an unrecoverable pipeline step fails."""


class VideoSkipped(Exception):
    """Raised when a video should be silently skipped (already archived, etc.)."""


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    """
    Run a subprocess, stream stderr to the logger, and optionally raise on failure.

    Args:
        cmd:   The command + arguments to execute.
        check: If True, raise PipelineError when the process exits non-zero.

    Returns:
        The completed process object (stdout/stderr as strings).
    """
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        snippet = result.stderr.strip()[:200]
        if check:
            raise PipelineError(f"Command failed ({result.returncode}): {snippet}")
        log.warning("Process note: %s", snippet)

    return result


# ---------------------------------------------------------------------------
# Channel loading
# ---------------------------------------------------------------------------


def load_channels() -> list[str]:
    """
    Return the list of YouTube channel handles / URLs from *channels.json*.
    If the file doesn't exist it is created with sensible defaults.
    """
    if not CHANNELS_FILE.exists():
        log.warning("%s not found — writing defaults and continuing.", CHANNELS_FILE)
        CHANNELS_FILE.write_text(json.dumps(DEFAULT_CHANNELS, indent=2))

    channels: list[str] = json.loads(CHANNELS_FILE.read_text())

    if not channels:
        raise PipelineError(f"{CHANNELS_FILE} is empty — add at least one channel.")

    return channels


# ---------------------------------------------------------------------------
# Video ID discovery
# ---------------------------------------------------------------------------


def fetch_video_ids(channel: str, limit: int) -> list[str]:
    """
    Use yt-dlp to list the *limit* most recent video IDs for *channel*.

    Args:
        channel: YouTube channel handle or URL (e.g. "@MKBHD").
        limit:   Maximum number of IDs to return.

    Returns:
        A (possibly empty) list of YouTube video ID strings.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--get-id",
        "--playlist-end", str(limit),
        "--no-warnings",
        channel,
    ]
    result = run(cmd)
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    log.info("Found %d video ID(s) for channel %s.", len(ids), channel)
    return ids


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _download_audio(video_id: str, dest: Path) -> Path:
    """
    Download the audio track of *video_id* as a WAV file into *dest*.

    Returns:
        Path to the downloaded .wav file.

    Raises:
        VideoSkipped: If yt-dlp reports the video is already in the archive
                      or the download otherwise produced no output file.
    """
    out_template = str(dest / f"{video_id}.wav")
    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--download-archive", str(ARCHIVE_FILE),
        "--output", out_template,
        "--no-playlist",
        "--no-warnings",
        url,
    ]
    run(cmd)

    wav = dest / f"{video_id}.wav"
    if not wav.exists():
        raise VideoSkipped(f"{video_id} — already archived or download produced no file.")

    return wav


def _slice_audio(wav: Path, chunk_dir: Path) -> int:
    """
    Split *wav* into fixed-length segments and write them to *chunk_dir*.

    Args:
        wav:       Source WAV file.
        chunk_dir: Destination directory for numbered segment files.

    Returns:
        Number of chunk files produced.

    Raises:
        PipelineError: If ffmpeg exits non-zero.
    """
    cmd = [
        "ffmpeg",
        "-i", str(wav),
        "-f", "segment",
        "-segment_time", str(SEGMENT_SECONDS),
        "-c", "copy",
        "-loglevel", "error",
        str(chunk_dir / "%03d.wav"),
    ]
    run(cmd, check=True)

    chunks = list(chunk_dir.glob("*.wav"))
    if not chunks:
        raise PipelineError(f"ffmpeg produced no chunks from {wav}.")

    return len(chunks)


def _write_kaggle_metadata(dest: Path) -> None:
    """Write the dataset-metadata.json required by the Kaggle API."""
    metadata = {
        "id": DATASET_ID,
        "title": DATASET_METADATA_TITLE,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dest / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))


def _upload_to_kaggle(api: KaggleApi, chunk_dir: Path, video_id: str) -> None:
    """
    Create a new Kaggle dataset version from *chunk_dir*.

    Args:
        api:       Authenticated KaggleApi instance.
        chunk_dir: Directory containing chunk WAVs + metadata.json.
        video_id:  Used only for the human-readable version message.

    Raises:
        PipelineError: Wraps any Kaggle SDK exception.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    version_notes = f"Auto-Harvest: {video_id} @ {timestamp}"

    try:
        api.dataset_create_version(
            str(chunk_dir),
            version_notes,
            dir_mode="zip",
            quiet=True,
        )
    except Exception as exc:
        raise PipelineError(f"Kaggle upload failed for {video_id}: {exc}") from exc


def process_video(video_id: str, api: KaggleApi) -> None:
    """
    Full pipeline for a single video: download → slice → upload.

    The entire working tree lives in a TemporaryDirectory that is
    cleaned up automatically on exit (even if an exception is raised).

    Args:
        video_id: 11-character YouTube video ID.
        api:      Authenticated KaggleApi instance.
    """
    log.info("▶  Processing  %s", video_id)

    with tempfile.TemporaryDirectory(prefix=f"palpha_{video_id}_") as tmp:
        work = Path(tmp)
        chunk_dir = work / "chunks"
        chunk_dir.mkdir()

        # 1 — Download
        try:
            wav = _download_audio(video_id, work)
        except VideoSkipped as exc:
            log.info("⏭  Skipping %s — %s", video_id, exc)
            return

        log.info("✔  Downloaded  %s (%.1f MB)", video_id, wav.stat().st_size / 1_048_576)

        # 2 — Slice
        n_chunks = _slice_audio(wav, chunk_dir)
        log.info("✔  Sliced into %d chunk(s) (%ds each)", n_chunks, SEGMENT_SECONDS)

        # 3 — Metadata
        _write_kaggle_metadata(chunk_dir)

        # 4 — Upload
        log.info("⬆  Uploading   %s to Kaggle …", video_id)
        _upload_to_kaggle(api, chunk_dir, video_id)
        log.info("✅ Done        %s", video_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("🚀 Pipeline-Alpha starting up …")

    # Validate required config up-front so failures are obvious immediately.
    if not DATASET_ID:
        raise PipelineError(
            "KAGGLE_DATASET environment variable is not set. "
            "Add it in Railway → Variables."
        )

    # Kaggle authentication
    try:
        api = KaggleApi()
        api.authenticate()
        log.info("🔑 Kaggle authenticated.")
    except Exception as exc:
        raise PipelineError(
            f"Kaggle authentication failed: {exc}. "
            "Check KAGGLE_USERNAME and KAGGLE_KEY in your Railway Variables."
        ) from exc

    # Channel selection
    channels = load_channels()
    target = random.choice(channels)
    log.info("🎯 Target channel: %s (pool size: %d)", target, len(channels))

    # Discover videos
    video_ids = fetch_video_ids(target, FETCH_LIMIT)
    if not video_ids:
        log.warning("No videos found for %s — nothing to do.", target)
        return

    log.info(
        "⚙  Processing %d video(s) with %d worker(s) …",
        len(video_ids), min(WORKERS, len(video_ids)),
    )

    # Process in parallel
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_video, vid, api): vid for vid in video_ids}
        for future in as_completed(futures):
            vid = futures[future]
            try:
                future.result()
            except PipelineError as exc:
                log.error("❌ %s failed: %s", vid, exc)
                errors.append(vid)
            except Exception as exc:
                log.exception("💥 Unexpected error for %s: %s", vid, exc)
                errors.append(vid)

    # Summary
    succeeded = len(video_ids) - len(errors)
    log.info(
        "🏁 Run complete — %d succeeded, %d failed.",
        succeeded, len(errors),
    )
    if errors:
        log.warning("Failed IDs: %s", ", ".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        log.critical("💀 Fatal: %s", exc)
        sys.exit(1)
