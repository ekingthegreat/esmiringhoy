import logging
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Dict, Optional
import yt_dlp

logger = logging.getLogger("clipforge.youtube")

YOUTUBE_URL_REGEX = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?.*v=|shorts/|embed/)|youtu\.be/)([\w-]{11})",
    re.IGNORECASE,
)
ANSI_ESCAPE_REGEX = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def get_js_runtimes_config() -> Dict[str, Any]:
    """Auto-detect available JS runtime (Node.js or Deno) to satisfy yt-dlp EJS requirements."""
    runtimes: Dict[str, Any] = {}
    if shutil.which("node"):
        runtimes["node"] = {}
    if shutil.which("deno"):
        runtimes["deno"] = {}
    return runtimes


def clean_yt_error_message(error: Any) -> str:
    """
    Clean up yt-dlp error messages by stripping ANSI color escapes and
    internal boilerplate, returning a user-friendly error string.
    """
    raw = ANSI_ESCAPE_REGEX.sub("", str(error)).strip()
    raw = re.sub(r"^ERROR:\s*(\[youtube\]\s*)?", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.split("; please report")[0].strip()
    raw = raw.split("; see ")[0].strip()

    lower_raw = raw.lower()
    if "private video" in lower_raw:
        return "This YouTube video is private and cannot be processed."
    if "unavailable" in lower_raw or "deleted" in lower_raw:
        return "This YouTube video is unavailable or has been removed."
    if "sign in" in lower_raw or "age" in lower_raw:
        return "This YouTube video requires sign-in or is age-restricted."
    if "403" in lower_raw or "forbidden" in lower_raw or "bot" in lower_raw:
        return f"YouTube blocked or throttled the download (HTTP 403 / anti-bot challenge). {raw}"
    if "timed out" in lower_raw or "timeout" in lower_raw:
        return f"Connection timed out while retrieving video from YouTube. {raw}"

    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return raw or "An unknown error occurred while downloading the YouTube video."


def is_valid_youtube_url(url: str) -> bool:
    """Check if the provided string matches valid YouTube URL patterns."""
    if not url or not isinstance(url, str):
        return False
    return bool(YOUTUBE_URL_REGEX.search(url.strip()))


def extract_youtube_info(url: str) -> Dict[str, Any]:
    """
    Validate and extract metadata for a YouTube video without downloading.
    Raises ValueError with a user-friendly message if the video is invalid, private, or unavailable.
    """
    clean_url = (url or "").strip()
    if not is_valid_youtube_url(clean_url):
        raise ValueError(
            "Invalid YouTube URL format. Please provide a standard YouTube video link "
            "(e.g., https://www.youtube.com/watch?v=... or https://youtu.be/...)."
        )

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "noplaylist": True,
        "skip_download": True,
        "socket_timeout": 20,
    }
    js_runtimes = get_js_runtimes_config()
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        cleaned_err = clean_yt_error_message(e)
        logger.warning(f"yt-dlp download error for {clean_url}: {cleaned_err}")
        raise ValueError(f"Could not retrieve YouTube video: {cleaned_err}")
    except Exception as e:
        logger.exception(f"Unexpected error validating YouTube URL: {clean_url}")
        raise ValueError(f"Failed to inspect YouTube video: {clean_yt_error_message(e)}")

    if not info:
        raise ValueError("Could not retrieve information for this YouTube video.")

    if info.get("is_live", False):
        raise ValueError("Live streams cannot be processed. Please use a recorded video.")

    duration = float(info.get("duration") or 0)
    if duration <= 0:
        raise ValueError("The video duration could not be determined or is zero.")
    if duration > 7200:  # 2 hours max
        raise ValueError(
            f"Video is too long ({int(duration // 60)} minutes). Maximum supported length is 120 minutes."
        )

    return {
        "video_id": info.get("id", ""),
        "title": info.get("title", "Untitled Video"),
        "duration": duration,
        "channel": info.get("channel") or info.get("uploader") or "Unknown Channel",
        "thumbnail": info.get("thumbnail", ""),
        "url": info.get("webpage_url", clean_url),
        "view_count": info.get("view_count", 0),
    }


def download_youtube_video(
    url: str,
    output_dir: Path,
    filename_prefix: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """
    Download video from YouTube using yt-dlp, preserving maximum practical quality
    (best video up to 1080p + best audio merged to MP4).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_mp4 = output_dir / f"{filename_prefix}.mp4"

    # In case previous attempt left a partial file
    if target_mp4.exists():
        target_mp4.unlink(missing_ok=True)

    def _progress_hook(d: Dict[str, Any]):
        if not progress_callback:
            return
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = min(100.0, max(0.0, (downloaded / total) * 100))
                progress_callback(pct, f"Downloading video stream: {int(pct)}%")
        elif status == "finished":
            progress_callback(100.0, "Download complete. Merging streams with FFmpeg...")

    js_runtimes = get_js_runtimes_config()
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl": str(output_dir / f"{filename_prefix}.%(ext)s"),
        "merge_output_format": "mp4",
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "noplaylist": True,
        "socket_timeout": 60,
        "retries": 10,
        "fragment_retries": 10,
        "file_access_retries": 5,
        "continuedl": True,
    }
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        # Clean up any leftover artifacts
        cleanup_temp_download_artifacts(output_dir, filename_prefix)
        cleaned_err = clean_yt_error_message(e)
        raise RuntimeError(f"YouTube download failed: {cleaned_err}")
    except Exception as e:
        cleanup_temp_download_artifacts(output_dir, filename_prefix)
        cleaned_err = clean_yt_error_message(e)
        raise RuntimeError(f"Unexpected error during YouTube download: {cleaned_err}")

    if not target_mp4.exists():
        # Check if downloaded with another extension that needs rename
        matches = list(output_dir.glob(f"{filename_prefix}.*"))
        for m in matches:
            if m.suffix.lower() in [".mp4", ".mkv", ".webm"]:
                return m
        cleanup_temp_download_artifacts(output_dir, filename_prefix)
        raise FileNotFoundError(f"Expected download file {target_mp4} was not found after retrieval.")

    return target_mp4


def cleanup_temp_download_artifacts(output_dir: Path, filename_prefix: str) -> None:
    """Remove any temporary files or partial downloads matching the prefix."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    for f in output_dir.glob(f"{filename_prefix}*"):
        try:
            if f.is_file():
                f.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not remove temp artifact {f}: {e}")
