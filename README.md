# ClipForge AI 🎬

ClipForge AI is an automated video studio that transforms YouTube videos into viral 9:16 vertical Shorts, Reels, and TikTok clips with AI face tracking, intelligent layout selection, and animated captions.

---

## 🚀 Features

- **Automated YouTube Video Retrieval**: Paste any YouTube URL (standard videos, Shorts, podcasts) and the system automatically extracts and downloads the video at original source quality up to 1080p using `yt-dlp`. No manual video downloads or file uploads required.
- **Stabilized Neural Face Tracking**: Tracks primary speakers across frames using OpenCV with:
  - **Deadband Filtering**: Eliminates micro-jitter and unnecessary camera movement from small head motions.
  - **Temporal Damping & EMA Smoothing**: Generates natural, cinematic ease-in/ease-out camera panning.
  - **Edge Protection**: Preserves facial features (forehead, chin, and sides) even when subjects move near the edges of a 16:9 frame.
  - **Gap Recovery**: Gracefully handles momentary detection failures by maintaining tracking state.
- **Professional 3-Section 9:16 Vertical Video Layout**:
  - **Top Blurred Background (8–12%)**: Generated synchronously from source frames, scaled, cropped, and smoothed with strong Gaussian blur and subtle darkening.
  - **Center Main Video (76–82%)**: Prominently centered, preserving source footage aspect ratio with zero distortion or stretching.
  - **Bottom Blurred Background (8–12%)**: Synchronized blurred frame matching the top section to eliminate awkward letterboxing, visible gaps, or black bars.
  - **Intelligent Layout Selection (Single-Subject vs. Stacked 2-View)**:
    - **Mode A (Single-Subject Tracking)**: When 1 subject or clustered subjects are present, smoothly tracks the primary speaker in the main video section.
    - **Mode B (Synchronized Stacked 2-View Layout)**: For dual-speaker podcast/interview setups, splits the main section into upper and lower views with a stylish divider line.
- **Bold Subtitle Design with Professional Glow Effect & Animations**:
  - **Bold Typography**: Modern bold sans-serif font (Arial Bold / Montserrat style) sized at 64–88px (default 74px at 1080×1920) for optimal mobile readability.
  - **Soft Outer Glow & Stroke**: Deep dark outline (4.5px), subtle drop shadow (2.5px), and a soft outer glow (`\blur2.2`) ensuring sharp, high-contrast readability over complex backgrounds.
  - **Emphasis Word Highlighting**: Configurable accent color (vibrant yellow/amber) highlights key content words while keeping the remaining text crisp white.
  - **Smooth Animations**: Restrained pop-in / scale-up (`\fscx92` to `100%`) and smooth fade-in (`\fad(100, 80)`) synchronized with dialogue pacing.
  - **Adaptive Face-Aware Placement**: Subtitles sit in the lower-middle portion of the main video section, safe above the bottom blurred section, and dynamically adjust upward when the speaker's face is positioned low in the frame.
  - **Smart Formatting**: Breaks speech into natural short phrases, balanced into at most 2 lines without clipping.
- **Automatic YouTube Shorts Title Generation (with "MrBeast" Keyword)**:
  - **Content-Driven Extraction**: Analyzes clip dialogue directly from the generated `.srt` file, transcript segments, or video metadata to extract the core action, key numbers/prizes, and peak moments.
  - **Viral MrBeast Style**: Formulates curiosity-driven, high-retention titles tailored for YouTube Shorts (e.g., *"MrBeast Gives Away Millions in This Insane Challenge"*, *"MrBeast Builds an Entire City for a Huge Challenge"*).
  - **Guaranteed Keyword Enforcement**: Validates that every generated title contains the exact keyword `"MrBeast"`.
  - **Unique Across Batch**: Guarantees distinct, non-repetitive title structures across multiple clips originating from the same source video.
  - **1-Click Copy & Inline Editing**: Results UI displays titles directly above each clip's video preview with a 1-click clipboard copy button and an editable input field.
- **Modern Responsive Web UI**: Paste-and-validate YouTube input, live video preview card, stage-by-stage progress tracking, in-browser video players, editable titles with 1-click copy, and 1-click downloads for videos, compilations, and `.srt` subtitle files.

---

## 🛠️ Prerequisites

1. **Python 3.10 - 3.13**
2. **FFmpeg & FFprobe**: Must be installed and accessible in your system `PATH` with `libass` support enabled.

---

## 🏁 Quick Start

### 1. Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Studio
```bash
python run.py
```
Or directly with Uvicorn:
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Open in Browser
Navigate to:
```
http://127.0.0.1:8000
```

---

## ⚙️ Configuration (`.env`)

You can customize behavior via the `.env` file:

```ini
WHISPER_MODEL=tiny           # Model options: tiny, base, small, medium, large
MAX_UPLOAD_SIZE_MB=2048      # Max video file upload size
DEFAULT_CLIP_COUNT=5         # Default clips generated per video
DEFAULT_SUBTITLE_STYLE=viral # Default subtitle style (viral, professional, minimal)
HOST=127.0.0.1
PORT=8000
```

---

## 🧪 Running Automated Tests

Run the full system integration test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_full_system.py
```

Run the standalone transcription test:

```powershell
.\.venv\Scripts\python.exe tests/test_transcription.py
```

---

## 📁 Project Architecture

```
clipforge/
├── app/
│   ├── main.py                     # FastAPI application & API routes (validate-url, process-url)
│   ├── services/
│   │   ├── youtube_service.py      # yt-dlp URL validation, metadata inspection & download
│   │   ├── face_tracking_service.py# OpenCV face detection, deadband tracking & layout selection
│   │   ├── video_service.py        # FFmpeg single/2-view rendering & transition compilation
│   │   ├── pipeline_service.py     # Background job queue & pipeline orchestrator
│   │   ├── transcription_service.py# Whisper transcription & audio analysis
│   │   ├── clipping_service.py     # Candidate scoring & highlight selection
│   │   └── subtitle_service.py     # SRT & layout-aware ASS subtitle generators
│   ├── static/
│   │   ├── css/style.css           # Modern dark-mode studio UI & layout badge styling
│   │   └── js/main.js              # URL paste/validation, polling, layout tags & compilation player
│   └── templates/
│       └── index.html              # Studio web interface with YouTube input & compilation card
├── models/                         # Local storage for neural model checkpoints
├── storage/
│   ├── uploads/                    # Downloaded and processed source videos
│   ├── clips/                      # Rendered 9:16 vertical MP4s, compilations & thumbnails
│   ├── subtitles/                  # Generated .srt & .ass files
│   └── temp/                       # Temporary processing scratchpads & job states
├── tests/
│   ├── test_full_system.py         # End-to-end integration tests (9 test suites)
│   └── test_transcription.py       # Standalone transcription test
├── .env                            # Environment configuration
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
└── run.py                          # Studio server launcher
```
"# esmiringhoy" 
