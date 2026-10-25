import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger("clipforge.transcription")

_model = None
_loaded_model_name = None

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def check_audio_stream(video_path: Path) -> bool:
    """Check if the video file contains an audio stream."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
        return len(data.get("streams", [])) > 0
    except Exception:
        return False


def get_model(model_name: Optional[str] = None):
    """Load or retrieve the cached Whisper model with corruption recovery."""
    global _model, _loaded_model_name
    import whisper

    selected_model = model_name or os.getenv("WHISPER_MODEL", "tiny")

    if _model is not None and _loaded_model_name == selected_model:
        return _model

    # Check for partially downloaded or corrupted model file
    model_file = MODELS_DIR / f"{selected_model}.pt"
    try:
        _model = whisper.load_model(selected_model, download_root=str(MODELS_DIR))
        _loaded_model_name = selected_model
    except RuntimeError as err:
        err_msg = str(err).lower()
        if "checksum does not" in err_msg or "corrupt" in err_msg or "eof" in err_msg:
            logger.warning(f"Corrupted model file detected for {selected_model}, removing and retrying...")
            if model_file.exists():
                model_file.unlink(missing_ok=True)
            _model = whisper.load_model(selected_model, download_root=str(MODELS_DIR))
            _loaded_model_name = selected_model
        else:
            raise

    return _model


def transcribe_video(
    video_path: Path,
    model_name: Optional[str] = None,
    word_timestamps: bool = True,
) -> Dict[str, Any]:
    """Transcribe a video using Whisper and return text and segment timestamps."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check for audio stream
    has_audio = check_audio_stream(video_path)
    if not has_audio:
        return {
            "text": "",
            "language": "en",
            "segments": [],
            "has_speech": False,
            "warning": "No audio stream detected in video file.",
        }

    model = get_model(model_name)

    # Run transcription with word timestamps enabled for precise subtitle timing
    try:
        result = model.transcribe(
            str(video_path),
            verbose=False,
            word_timestamps=word_timestamps,
        )
    except Exception as e:
        logger.warning(f"Word timestamps failed, falling back to segment transcription: {e}")
        result = model.transcribe(
            str(video_path),
            verbose=False,
            word_timestamps=False,
        )

    segments_out: List[Dict[str, Any]] = []
    for idx, seg in enumerate(result.get("segments", [])):
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue

        words_data = []
        for w in seg.get("words", []):
            words_data.append({
                "word": w.get("word", "").strip(),
                "start": float(w.get("start", 0)),
                "end": float(w.get("end", 0)),
            })

        segments_out.append({
            "id": idx,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg_text,
            "words": words_data,
        })

    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language", "en"),
        "segments": segments_out,
        "has_speech": len(segments_out) > 0,
    }