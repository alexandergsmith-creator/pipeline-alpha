#!/usr/bin/env python3
"""
Pipeline Alpha — YouTube Audio Harvest → Kaggle Upload
=======================================================
Pulls audio from YouTube, slices into 10-minute chunks,
and versions them into a Kaggle dataset. 
"""

from __future__ import annotations

if True: # Formatting block for clean imports
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
    from kaggle.api.kaggle_api_extended import KaggleApi

# ---------------------------------------------------------------------------
# Configuration (Railway Variables)
# ---------------------------------------------------------------------------
DATASET_ID = os.environ.get("KAGGLE_DATASET", "")
WORKERS = int(os.environ.get("WORKERS", "1"))
FETCH_LIMIT = int(os.environ.get("FETCH_LIMIT", "1"))
SEGMENT_SECONDS = int(os.environ.get("SEGMENT_SECONDS", "600"))

CHANNELS_FILE = Path("channels.json")
ARCHIVE_FILE = Path("archive.txt")
DEFAULT_CHANNELS = ["https://www.youtube.com/@MKBHD/videos"]

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline-alpha")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], check: bool = False):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        snippet = result.stderr.strip()[:200]
        if check:
            raise RuntimeError(f"Command failed: {snippet}")
        log.warning("System Note: %s", snippet)
    return result

def _download_audio(video_id: str, dest: Path) -> Path:
    out_template = str(dest / f"{video_id}.wav")
    url = f"https://www.youtube.com/watch?v={video_id}"

    # The "Android" client argument helps bypass bot detection on cloud IPs
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--extractor-args", "youtube:player-client=android,web",
        "--download-archive", str(ARCHIVE_FILE),
        "--output", out_template,
        "--no-playlist",
        "--no-warnings",
        url,
    ]
    run(cmd)

    wav = dest / f"{video_id}.wav"
    if not wav.exists():
        raise FileNotFoundError(f"Download failed for {video_id}")
    return wav

def _slice_audio(wav: Path, chunk_dir: Path):
    cmd = [
        "ffmpeg", "-i", str(wav),
        "-f", "segment", "-segment_time", str(SEGMENT_SECONDS),
        "-c", "copy", "-loglevel", "error",
        str(chunk_dir / "%03d.wav"),
    ]
    run(cmd, check=True)
    return len(list(chunk_dir.glob("*.wav")))

def _upload_to_kaggle(api: KaggleApi, chunk_dir: Path, video_id: str):
    meta = {
        "id": DATASET_ID,
        "title": "Pipeline Alpha Harvest",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (chunk_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    api.dataset_create_version(
        str(chunk_dir),
        f"Auto-Harvest: {video_id} @ {timestamp}",
        dir_mode="zip",
        quiet=True,
    )

def process_video(video_id: str, api: KaggleApi):
    log.info("▶  Processing  %s", video_id)
    with tempfile.TemporaryDirectory(prefix=f"palpha_{video_id}_") as tmp:
        work = Path(tmp)
        chunk_dir = work / "chunks"
        chunk_dir.mkdir()

        try:
            wav = _download_audio(video_id, work)
            log.info("✔  Downloaded  %s (%.1f MB)", video_id, wav.stat().st_size / 1_048_576)
            
            n_chunks = _slice_audio(wav, chunk_dir)
            log.info("✔  Sliced into %d chunk(s)", n_chunks)

            log.info("⬆  Uploading   %s to Kaggle...", video_id)
            _upload_to_kaggle(api, chunk_dir, video_id)
            log.info("✅ Done         %s", video_id)
        except Exception as e:
            log.error("❌ Failed %s: %s", video_id, e)

# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------

def main():
    log.info("🚀 Pipeline-Alpha starting up...")

    if not DATASET_ID:
        log.error("KAGGLE_DATASET variable missing.")
        return

    try:
        api = KaggleApi()
        api.authenticate()
        log.info("🔑 Kaggle authenticated.")
    except Exception as e:
        log.error("Kaggle Auth Failed: %s", e)
        return

    if not CHANNELS_FILE.exists():
        CHANNELS_FILE.write_text(json.dumps(DEFAULT_CHANNELS))

    channels = json.loads(CHANNELS_FILE.read_text())
    target = random.choice(channels)
    
    # Discovery
    cmd = ["yt-dlp", "--flat-playlist", "--get-id", "--playlist-end", str(FETCH_LIMIT), target]
    video_ids = [line.strip() for line in run(cmd).stdout.splitlines() if line.strip()]

    if not video_ids:
        log.warning("No videos found for %s.", target)
        return

    log.info("⚙  Processing %d video(s) with %d worker(s)...", len(video_ids), WORKERS)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(process_video, vid, api) for vid in video_ids]
        for f in as_completed(futures):
            pass

    log.info("🏁 Run complete.")

if __name__ == "__main__":
    main()
