import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import uuid

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.services.pipeline_service import (
    create_job,
    get_job_state,
    process_video_job,
    update_clip_title_in_job,
)
from app.services.video_service import get_video_info
from app.services.youtube_service import (
    extract_youtube_info,
    is_valid_youtube_url,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

STORAGE_DIR = PROJECT_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
CLIPS_DIR = STORAGE_DIR / "clips"
SUBTITLES_DIR = STORAGE_DIR / "subtitles"
TEMP_DIR = STORAGE_DIR / "temp"

for d in [UPLOAD_DIR, CLIPS_DIR, SUBTITLES_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
}
MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB

# Thread pool for CPU-bound video & transcription processing
pipeline_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="clipforge-worker")

app = FastAPI(
    title="ClipForge AI",
    description="Local AI video studio for turning YouTube videos into viral vertical shorts.",
    version="2.0.0",
)

# CORS middleware for local API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend assets
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

# Mount generated clips and subtitles for direct streaming and downloading
app.mount(
    "/storage/clips",
    StaticFiles(directory=CLIPS_DIR),
    name="clips",
)

app.mount(
    "/storage/subtitles",
    StaticFiles(directory=SUBTITLES_DIR),
    name="subtitles",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


class ValidateUrlRequest(BaseModel):
    url: str


class ProcessUrlRequest(BaseModel):
    url: str
    clip_count: int = Field(default=5)
    subtitle_style: str = Field(default="viral")
    transition_style: str = Field(default="fade")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = BASE_DIR / "static" / "img" / "logo.png"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon not found")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "ClipForge AI",
        "version": "2.0.0",
    }


@app.post("/api/validate-url")
async def validate_url(payload: ValidateUrlRequest):
    """
    Validate a YouTube URL and extract video title, thumbnail, and duration
    without downloading the video.
    """
    url = (payload.url or "").strip()
    if not is_valid_youtube_url(url):
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL format. Please enter a valid YouTube video link.",
        )

    try:
        info = extract_youtube_info(url)
        return {
            "status": "ok",
            "valid": True,
            **info,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect YouTube video: {str(e)}")


@app.post("/api/process-url")
async def process_youtube_url(
    payload: ProcessUrlRequest,
    background_tasks: BackgroundTasks,
):
    """
    Queue a YouTube video for automated retrieval, transcription, face tracking,
    vertical re-framing, and subtitle burning.
    """
    url = (payload.url or "").strip()
    if not is_valid_youtube_url(url):
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL format. Please provide a standard YouTube video link.",
        )

    if payload.clip_count not in {3, 5, 10, 20}:
        raise HTTPException(
            status_code=400,
            detail="Invalid clip count. Choose 3, 5, 10, or 20.",
        )

    if payload.subtitle_style not in {"viral", "professional", "minimal"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid subtitle style. Choose viral, professional, or minimal.",
        )

    if payload.transition_style not in {"fade", "slideleft", "zoomin", "smoothleft", "cut"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid transition style.",
        )

    # Initialize job in pipeline
    job_id = create_job(
        youtube_url=url,
        clip_count=payload.clip_count,
        subtitle_style=payload.subtitle_style,
        transition_style=payload.transition_style,
    )

    # Queue background task to process the job asynchronously in executor
    background_tasks.add_task(pipeline_executor.submit, process_video_job, job_id)

    return {
        "status": "queued",
        "job_id": job_id,
        "url": url,
        "clip_count": payload.clip_count,
        "subtitle_style": payload.subtitle_style,
        "transition_style": payload.transition_style,
        "message": "YouTube video queued for processing.",
    }


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    clip_count: int = Form(5),
    subtitle_style: str = Form("viral"),
    transition_style: str = Form("fade"),
):
    """Fallback manual upload endpoint for testing or local files."""
    extension = Path(video.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{extension}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    if clip_count not in {3, 5, 10, 20}:
        raise HTTPException(
            status_code=400,
            detail="Invalid clip count. Choose 3, 5, 10, or 20.",
        )
    if subtitle_style not in {"viral", "professional", "minimal"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid subtitle style. Choose viral, professional, or minimal.",
        )

    filename = f"{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_DIR / filename
    total_size = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 2 GB upload limit.",
                    )
                output.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as e:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to write file: {str(e)}")
    finally:
        await video.close()

    try:
        get_video_info(destination)
    except Exception as e:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file is not a valid video or cannot be decoded: {str(e)}",
        )

    job_id = create_job(
        video_filename=filename,
        clip_count=clip_count,
        subtitle_style=subtitle_style,
        transition_style=transition_style,
    )

    background_tasks.add_task(pipeline_executor.submit, process_video_job, job_id)

    return {
        "status": "queued",
        "job_id": job_id,
        "filename": filename,
        "clip_count": clip_count,
        "subtitle_style": subtitle_style,
        "message": "Video uploaded successfully. Processing started.",
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = get_job_state(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


class UpdateTitleRequest(BaseModel):
    title: str


@app.post("/api/jobs/{job_id}/clips/{clip_id}/title")
async def update_title(job_id: str, clip_id: str, payload: UpdateTitleRequest):
    new_title = (payload.title or "").strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    success = update_clip_title_in_job(job_id, clip_id, new_title)
    if not success:
        raise HTTPException(status_code=404, detail="Job or clip not found.")

    return {
        "status": "ok",
        "job_id": job_id,
        "clip_id": clip_id,
        "title": new_title,
    }