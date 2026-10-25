from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.services.transcription_service import transcribe_video


def run_transcription_test():
    video_path = PROJECT_DIR / "storage" / "temp" / "test_synth.mp4"
    if not video_path.exists():
        video_path.parent.mkdir(parents=True, exist_ok=True)
        # Generate short test clip with audio
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=3",
            "-c:v", "libx264", "-c:a", "aac",
            str(video_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    print(f"Transcribing: {video_path}")
    result = transcribe_video(video_path)
    print("Transcription result:", result)
    print("Test passed successfully.")


if __name__ == "__main__":
    run_transcription_test()