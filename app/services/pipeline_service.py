import asyncio
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

from app.services.clipping_service import generate_candidate_clips
from app.services.face_tracking_service import face_tracker
from app.services.subtitle_service import create_ass, create_srt
from app.services.title_service import generate_clip_title
from app.services.transcription_service import transcribe_video
from app.services.video_service import (
    create_stitched_compilation,
    create_vertical_clip,
    extract_thumbnail,
    get_video_info,
)
from app.services.youtube_service import (
    cleanup_temp_download_artifacts,
    download_youtube_video,
    extract_youtube_info,
)

logger = logging.getLogger("clipforge.pipeline")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = PROJECT_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
CLIPS_DIR = STORAGE_DIR / "clips"
SUBTITLES_DIR = STORAGE_DIR / "subtitles"
TEMP_DIR = STORAGE_DIR / "temp"
JOBS_DIR = TEMP_DIR / "jobs"

for d in [UPLOAD_DIR, CLIPS_DIR, SUBTITLES_DIR, TEMP_DIR, JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# In-memory job registry protected by a lock
_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def save_job_state(job_id: str, data: Dict[str, Any]) -> None:
    """Save job state to memory and persist to disk."""
    with _jobs_lock:
        _jobs[job_id] = data

    job_file = JOBS_DIR / f"{job_id}.json"
    try:
        job_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not persist job {job_id} to disk: {e}")


def get_job_state(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve job state from memory or disk."""
    with _jobs_lock:
        if job_id in _jobs:
            return _jobs[job_id]

    job_file = JOBS_DIR / f"{job_id}.json"
    if job_file.exists():
        try:
            data = json.loads(job_file.read_text(encoding="utf-8"))
            with _jobs_lock:
                _jobs[job_id] = data
            return data
        except Exception:
            return None
    return None


def create_job(
    youtube_url: Optional[str] = None,
    video_filename: Optional[str] = None,
    clip_count: int = 5,
    subtitle_style: str = "viral",
    transition_style: str = "fade",
) -> str:
    """Initialize and register a new processing job for YouTube URL or local file."""
    # Backward compatibility: if first positional argument is a filename instead of URL
    if youtube_url and not str(youtube_url).lower().startswith("http") and ("." in str(youtube_url) or not video_filename):
        video_filename = youtube_url
        youtube_url = None

    job_id = uuid.uuid4().hex
    initial_data = {
        "job_id": job_id,
        "youtube_url": youtube_url,
        "video_filename": video_filename,
        "clip_count": clip_count,
        "subtitle_style": subtitle_style,
        "transition_style": transition_style,
        "status": "queued",
        "progress": 0,
        "current_stage": "Job queued in studio",
        "created_at": time.time(),
        "video_metadata": {},
        "clips": [],
        "compilation": None,
        "error": None,
    }
    save_job_state(job_id, initial_data)
    return job_id


def process_video_job(job_id: str) -> None:
    """
    Synchronous worker pipeline executed in a background thread.
    Carries out:
    1. YouTube Retrieval & Validation (or Local File check)
    2. Audio Transcription (Whisper)
    3. Engagement Candidate Selection
    4. Face Tracking & Layout Selection (Single vs Two-View Stacked)
    5. Vertical 9:16 Video Rendering & Styled Captions
    6. Multi-Clip Compilation Reel with Transitions
    """
    job = get_job_state(job_id)
    if not job:
        logger.error(f"Job {job_id} not found.")
        return

    youtube_url = job.get("youtube_url")
    video_filename = job.get("video_filename")
    clip_count = int(job.get("clip_count", 5))
    subtitle_style = job.get("subtitle_style", "viral")
    transition_style = job.get("transition_style", "fade")

    video_path: Optional[Path] = None
    is_temp_download = False

    try:
        # Step 1: Video Source Acquisition (0% -> 25%)
        job["status"] = "processing"
        job["progress"] = 5

        if youtube_url:
            job["current_stage"] = "Validating YouTube URL and inspecting metadata..."
            save_job_state(job_id, job)

            yt_info = extract_youtube_info(youtube_url)
            job["video_metadata"] = yt_info
            job["progress"] = 10
            job["current_stage"] = f"Retrieving video: {yt_info.get('title', 'YouTube Video')}..."
            save_job_state(job_id, job)

            def _dl_progress(percent: float, stage_text: str):
                # Map download progress from 10% to 25%
                job["progress"] = 10 + int((percent / 100.0) * 15)
                job["current_stage"] = stage_text
                save_job_state(job_id, job)

            video_path = download_youtube_video(
                url=youtube_url,
                output_dir=UPLOAD_DIR,
                filename_prefix=job_id,
                progress_callback=_dl_progress,
            )
            is_temp_download = True
            job["video_filename"] = video_path.name
        elif video_filename:
            video_path = UPLOAD_DIR / video_filename
        else:
            raise ValueError("No video source provided (neither YouTube URL nor filename).")

        if not video_path.exists():
            raise FileNotFoundError(f"Source video file {video_path} could not be found.")

        # Step 2: Verify video properties (25%)
        job["progress"] = 25
        job["current_stage"] = "Analyzing video properties and audio channels..."
        save_job_state(job_id, job)

        video_info = get_video_info(video_path)
        total_duration = video_info["duration"]

        # Step 3: Transcribe video audio with Whisper (35%)
        job["progress"] = 35
        job["current_stage"] = "Transcribing speech using AI Whisper model..."
        save_job_state(job_id, job)

        transcription = transcribe_video(video_path)
        segments = transcription.get("segments", [])

        # Step 4: Highlight Candidate Selection (50%)
        job["progress"] = 50
        job["current_stage"] = "Selecting high-impact moments and highlights..."
        save_job_state(job_id, job)

        candidate_clips = generate_candidate_clips(
            segments=segments,
            total_duration=total_duration,
            clip_count=clip_count,
            target_duration=35,
            minimum_duration=15,
            maximum_duration=60,
        )

        if not candidate_clips:
            raise RuntimeError("Could not find suitable clip segments from video.")

        # Step 5: Face Tracking, Framing, Rendering & Caption Burning (50% -> 90%)
        generated_clips = []
        clip_paths: List[Path] = []
        total_clips = len(candidate_clips)
        used_titles: Set[str] = set()

        for idx, clip in enumerate(candidate_clips, 1):
            clip_id = f"{job_id}_clip_{idx}"
            clip_start = clip["start"]
            clip_end = clip["end"]
            clip_sub_segments = clip.get("segments", segments)

            job["current_stage"] = f"Face tracking & framing clip {idx} of {total_clips}..."
            base_pct = 50 + int(((idx - 1) / total_clips) * 38)
            job["progress"] = base_pct
            save_job_state(job_id, job)

            # Analyze face tracking and determine layout mode (Single vs Two-View)
            tracking_info = face_tracker.analyze_clip(
                video_path=video_path,
                start_time=clip_start,
                end_time=clip_end,
                sample_fps=3.0,
            )
            layout_mode = tracking_info.get("layout_mode", "single")

            job["current_stage"] = (
                f"Rendering clip {idx} ({'Stacked 2-View' if layout_mode == 'two_view' else 'Single Subject'}) with subtitles..."
            )
            job["progress"] = base_pct + int(38 / (total_clips * 2))
            save_job_state(job_id, job)

            # Generate subtitles (.srt and .ass) with layout-aware vertical alignment
            srt_path = SUBTITLES_DIR / f"{clip_id}.srt"
            ass_path = SUBTITLES_DIR / f"{clip_id}.ass"

            create_srt(
                segments=clip_sub_segments,
                output_path=srt_path,
                clip_start=clip_start,
                clip_end=clip_end,
                style=subtitle_style,
            )

            create_ass(
                segments=clip_sub_segments,
                output_path=ass_path,
                clip_start=clip_start,
                clip_end=clip_end,
                style=subtitle_style,
                layout_mode=layout_mode,
                tracking_info=tracking_info,
            )

            # Generate engaging unique YouTube Shorts title containing mandatory 'MrBeast'
            clip_title = generate_clip_title(
                clip_text=clip.get("text", ""),
                srt_path=srt_path,
                clip_index=idx,
                total_clips=total_clips,
                used_titles=used_titles,
                source_title=job.get("video_metadata", {}).get("title", ""),
            )

            # Render 9:16 vertical clip with intelligent framing and burned captions
            output_clip_path = CLIPS_DIR / f"{clip_id}.mp4"
            create_vertical_clip(
                input_path=video_path,
                output_path=output_clip_path,
                start=clip_start,
                end=clip_end,
                subtitle_path=ass_path,
                tracking_info=tracking_info,
            )

            # Extract thumbnail
            thumb_path = CLIPS_DIR / f"{clip_id}_thumb.jpg"
            extract_thumbnail(
                video_path=output_clip_path,
                output_image_path=thumb_path,
                timestamp=min(1.5, (clip_end - clip_start) / 2),
            )

            clip_paths.append(output_clip_path)
            generated_clips.append({
                "clip_id": clip_id,
                "clip_number": idx,
                "title": clip_title,
                "start": clip_start,
                "end": clip_end,
                "duration": round(clip_end - clip_start, 2),
                "layout_mode": layout_mode,
                "layout_label": "Two-View Split" if layout_mode == "two_view" else "Single Subject",
                "layout_description": tracking_info.get("description", ""),
                "text": clip.get("text", f"Short Highlight #{idx}"),
                "video_url": f"/storage/clips/{clip_id}.mp4",
                "thumbnail_url": f"/storage/clips/{clip_id}_thumb.jpg",
                "srt_url": f"/storage/subtitles/{clip_id}.srt",
                "score": clip.get("score", 1.0),
            })

        # Step 6: Create Stitched Compilation Reel with Transitions (90% -> 96%)
        compilation_data = None
        if len(clip_paths) >= 2:
            job["progress"] = 92
            job["current_stage"] = f"Assembling compilation reel with '{transition_style}' transitions..."
            save_job_state(job_id, job)

            compilation_path = CLIPS_DIR / f"{job_id}_compilation.mp4"
            try:
                create_stitched_compilation(
                    clip_paths=clip_paths,
                    output_path=compilation_path,
                    transition_type=transition_style,
                    transition_duration=0.4,
                )
                comp_thumb = CLIPS_DIR / f"{job_id}_compilation_thumb.jpg"
                extract_thumbnail(compilation_path, comp_thumb, timestamp=1.0)
                compilation_data = {
                    "video_url": f"/storage/clips/{job_id}_compilation.mp4",
                    "thumbnail_url": f"/storage/clips/{job_id}_compilation_thumb.jpg",
                    "clip_count": len(clip_paths),
                    "transition": transition_style,
                }
            except Exception as e:
                logger.warning(f"Could not build stitched compilation: {e}")

        # Step 7: Completed! (100%)
        job["status"] = "completed"
        job["progress"] = 100
        job["current_stage"] = f"Finished! Generated {len(generated_clips)} vertical clips."
        job["clips"] = generated_clips
        job["compilation"] = compilation_data
        save_job_state(job_id, job)
        logger.info(f"Job {job_id} successfully completed.")

    except Exception as e:
        logger.exception(f"Error processing job {job_id}: {e}")
        job["status"] = "failed"
        job["current_stage"] = "Processing failed"
        job["error"] = str(e)
        save_job_state(job_id, job)

    finally:
        # Cleanup temporary source files if downloaded from YouTube
        if is_temp_download and video_path and video_path.exists():
            cleanup_temp_download_artifacts(UPLOAD_DIR, job_id)


def update_clip_title_in_job(job_id: str, clip_id: str, new_title: str) -> bool:
    """Update and persist the title of a specific clip within a job."""
    from app.services.title_service import ensure_mrbeast_title

    job = get_job_state(job_id)
    if not job:
        return False

    valid_title = ensure_mrbeast_title(new_title)
    updated = False

    for clip in job.get("clips", []):
        if clip.get("clip_id") == clip_id:
            clip["title"] = valid_title
            updated = True
            break

    if updated:
        save_job_state(job_id, job)

    return updated
