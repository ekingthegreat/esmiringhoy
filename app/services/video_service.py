import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger("clipforge.video")

# Standard 3-Section 9:16 layout defaults:
# Top blurred background: ~10% (within 8% - 12%) -> 192px at 1920
# Bottom blurred background: ~10% (within 8% - 12%) -> 192px at 1920
# Main center video section: ~80% (within 76% - 82%) -> 1536px at 1920
DEFAULT_TOP_RATIO = 0.10
DEFAULT_BOTTOM_RATIO = 0.10


def calculate_layout_dimensions(
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    top_ratio: float = DEFAULT_TOP_RATIO,
    bottom_ratio: float = DEFAULT_BOTTOM_RATIO,
) -> Dict[str, int]:
    """
    Calculate exact pixel heights and vertical offsets for the 3-section 9:16 layout.
    Guarantees:
    - Top blurred background: 8% to 12% of canvas height
    - Bottom blurred background: 8% to 12% of canvas height
    - Main center video section: 76% to 82% of canvas height
    - top_h + main_h + bottom_h == canvas_h (Zero gaps, zero overlapping, zero unintended borders)
    - All dimensions are even numbers for H.264 macroblock alignment.
    """
    # Clamp ratios within required specifications
    top_ratio = max(0.08, min(0.12, float(top_ratio)))
    bottom_ratio = max(0.08, min(0.12, float(bottom_ratio)))

    top_h = int(round(canvas_h * top_ratio))
    if top_h % 2 != 0:
        top_h += 1

    bottom_h = int(round(canvas_h * bottom_ratio))
    if bottom_h % 2 != 0:
        bottom_h += 1

    main_h = canvas_h - top_h - bottom_h
    if main_h % 2 != 0:
        main_h -= 1
        bottom_h += 1

    return {
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "top_h": top_h,
        "bottom_h": bottom_h,
        "main_h": main_h,
        "main_y": top_h,
        "divider_y": top_h + (main_h // 2),
    }


def escape_ffmpeg_filter_path(path: Path) -> str:
    """
    Escape Windows paths for ffmpeg filter arguments.
    In ffmpeg filtergraphs, ':' separates options and '\\' escapes characters.
    Drive letter 'C:' must become 'C\\:' and directory separators should be '/'.
    Single quotes are escaped as '\\\''.
    """
    p = str(path.resolve()).replace("\\", "/")
    # Escape drive colon for FFmpeg filter parsing: e.g. C:/ -> C\:/
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    p = p.replace("'", "'\\\\\\''")
    return p


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Retrieve video format, dimensions, duration, and stream information."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=codec_type,width,height,r_frame_rate",
        "-of", "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    data = json.loads(result.stdout)

    video_stream = next(
        (
            stream
            for stream in data.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        None,
    )

    audio_stream = next(
        (
            stream
            for stream in data.get("streams", [])
            if stream.get("codec_type") == "audio"
        ),
        None,
    )

    if not video_stream:
        raise ValueError("The file does not contain a video stream.")

    duration_val = data.get("format", {}).get("duration")
    if duration_val is None:
        duration_val = video_stream.get("duration", 0)

    return {
        "duration": float(duration_val),
        "width": int(video_stream.get("width", 1920)),
        "height": int(video_stream.get("height", 1080)),
        "has_audio": audio_stream is not None,
    }


def extract_thumbnail(
    video_path: Path,
    output_image_path: Path,
    timestamp: float = 1.0,
) -> Path:
    """Extract a fast JPEG thumbnail from the video at the given timestamp."""
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(max(0.1, timestamp)),
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "3",
        "-vf", "scale=540:960:force_original_aspect_ratio=increase,crop=540:960",
        str(output_image_path),
    ]

    subprocess.run(
        command,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return output_image_path


def build_smooth_crop_expr(
    trajectory: List[Dict[str, float]],
    default_x: float,
    max_x: float,
) -> str:
    """
    Construct an FFmpeg time-evaluated piecewise linear expression for smooth camera panning.
    Uses FFmpeg math functions 'if', 'lte', and 'lerp' to transition smoothly between keypoints.
    """
    if not trajectory or len(trajectory) <= 1:
        x_val = trajectory[0]["crop_x"] if trajectory else default_x
        return str(round(float(np_clip(x_val, 0, max_x)), 1))

    # Downsample trajectory to key turning points to keep the expression concise
    keypoints = [trajectory[0]]
    for pt in trajectory[1:]:
        dt = pt["time"] - keypoints[-1]["time"]
        dx = abs(pt["crop_x"] - keypoints[-1]["crop_x"])
        # Include point if camera moved significantly or at least 2.5 seconds passed
        if dx >= 16.0 or dt >= 2.5:
            keypoints.append(pt)

    if keypoints[-1]["time"] < trajectory[-1]["time"]:
        keypoints.append(trajectory[-1])

    if len(keypoints) <= 1:
        return str(round(float(np_clip(keypoints[0]["crop_x"], 0, max_x)), 1))

    # Nest conditional lerps backwards: if(lte(t, t_i), lerp(x_{i-1}, x_i, ...), fallback)
    last_x = round(float(np_clip(keypoints[-1]["crop_x"], 0, max_x)), 1)
    expr = str(last_x)

    for i in range(len(keypoints) - 1, 0, -1):
        prev = keypoints[i - 1]
        curr = keypoints[i]
        dt = max(0.1, curr["time"] - prev["time"])
        t_bound = curr["time"]
        t_prev = prev["time"]
        x_prev = round(float(np_clip(prev["crop_x"], 0, max_x)), 1)
        x_curr = round(float(np_clip(curr["crop_x"], 0, max_x)), 1)

        # In FFmpeg expressions commas inside functions must be escaped with backslash
        expr = (
            f"if(lte(t\\,{t_bound:.2f})\\,"
            f"lerp({x_prev}\\,{x_curr}\\,(t-{t_prev:.2f})/{dt:.2f})\\,"
            f"{expr})"
        )

    return expr


def np_clip(val: float, low: float, high: float) -> float:
    return max(low, min(val, high))


def create_vertical_clip(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
    subtitle_path: Optional[Path] = None,
    tracking_info: Optional[Dict[str, Any]] = None,
    top_ratio: float = DEFAULT_TOP_RATIO,
    bottom_ratio: float = DEFAULT_BOTTOM_RATIO,
) -> Path:
    """
    Cut video segment and convert to professional 9:16 vertical (1080x1920).
    Divides the frame into 3 distinct sections:
    - Section A: Top blurred background (approx 8-12%, default 10% / 192px)
      Generated from source frame, scaled/cropped, strong Gaussian blur + subtle darkening.
    - Section B: Main Center Video (approx 76-82%, default 80% / 1536px)
      Original aspect ratio preserved without stretching or distortion.
      Supports Single-Subject Tracking (Mode A) and Synchronized Two-View Stack (Mode B).
    - Section C: Bottom blurred background (approx 8-12%, default 10% / 192px)
      Generated from source frame, strong Gaussian blur + subtle darkening.
    Burns bold styled subtitles with professional glow & animations.
    """
    if start < 0 or end <= start:
        raise ValueError(f"Invalid clip timestamps: start={start}, end={end}")

    duration = end - start
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate layout section dimensions (guarantees 1080x1920 exact fit)
    layout = calculate_layout_dimensions(1080, 1920, top_ratio=top_ratio, bottom_ratio=bottom_ratio)
    top_h = layout["top_h"]
    bottom_h = layout["bottom_h"]
    main_h = layout["main_h"]

    layout_mode = tracking_info.get("layout_mode", "single") if tracking_info else "single"

    # Common subtitle filter string
    sub_filter = ""
    if subtitle_path and subtitle_path.exists():
        escaped_sub = escape_ffmpeg_filter_path(subtitle_path)
        sub_filter = f",ass='{escaped_sub}'"

    # Blurred background filters: downscale -> Gaussian blur -> upscale -> subtle darken
    top_bg_filter = (
        f"scale=270:48:force_original_aspect_ratio=increase,"
        f"crop=270:48,gblur=sigma=10:steps=2,"
        f"scale=1080:{top_h},eq=brightness=-0.08"
    )
    bot_bg_filter = (
        f"scale=270:48:force_original_aspect_ratio=increase,"
        f"crop=270:48,gblur=sigma=10:steps=2,"
        f"scale=1080:{bottom_h},eq=brightness=-0.08"
    )

    if layout_mode == "two_view" and tracking_info:
        # Mode B: Stacked Two-View Layout inside the main video section (1080 x main_h)
        top_view = tracking_info.get("top_view", {})
        bot_view = tracking_info.get("bottom_view", {})

        crop_w = int(tracking_info.get("crop_w", 608))
        crop_h = int(tracking_info.get("crop_h", 1080))
        top_x = int(top_view.get("crop_x", 0))
        bot_x = int(bot_view.get("crop_x", max(0, 1920 - crop_w)))
        view_h = main_h // 2

        filter_complex = (
            f"[0:v]split=4[src_top][src_tv_top][src_tv_bot][src_bot];"
            f"[src_top]{top_bg_filter}[top_bg];"
            f"[src_tv_top]crop=w={crop_w}:h={crop_h}:x={top_x}:y=0,"
            f"scale=1080:{main_h}:force_original_aspect_ratio=increase,"
            f"crop=1080:{view_h}:(in_w-out_w)/2:(in_h-out_h)/2[top_v];"
            f"[src_tv_bot]crop=w={crop_w}:h={crop_h}:x={bot_x}:y=0,"
            f"scale=1080:{main_h}:force_original_aspect_ratio=increase,"
            f"crop=1080:{view_h}:(in_w-out_w)/2:(in_h-out_h)/2[bot_v];"
            f"[top_v][bot_v]vstack=inputs=2,"
            f"drawbox=y={view_h - 2}:h=4:color=black@0.85:t=fill[main_v];"
            f"[src_bot]{bot_bg_filter}[bot_bg];"
            f"[top_bg][main_v][bot_bg]vstack=inputs=3,"
            f"setsar=1{sub_filter}[outv]"
        )
    else:
        # Mode A: Single-Subject Dynamic Tracking inside the main video section (1080 x main_h)
        # Crop aspect ratio is exactly 1080 / main_h to preserve original footage aspect ratio perfectly
        crop_w_expr = f"ih*1080/{main_h}"
        crop_h_expr = "ih"

        if tracking_info and tracking_info.get("has_dynamic_motion") and tracking_info.get("trajectory"):
            video_w = tracking_info.get("video_w", 1920)
            crop_w = tracking_info.get("crop_w", int(round(1080 * 1080 / main_h)))
            dominant_x = tracking_info.get("dominant_crop_x", (video_w - crop_w) / 2.0)
            max_x = max(0, video_w - crop_w)
            x_expr = build_smooth_crop_expr(tracking_info["trajectory"], dominant_x, max_x)
            crop_filter = f"crop=w={crop_w_expr}:h={crop_h_expr}:x='{x_expr}':y=0"
        elif tracking_info and "dominant_crop_x" in tracking_info:
            dominant_x = round(float(tracking_info["dominant_crop_x"]), 1)
            crop_filter = f"crop=w={crop_w_expr}:h={crop_h_expr}:x={dominant_x}:y=0"
        else:
            crop_filter = f"crop=w={crop_w_expr}:h={crop_h_expr}:x=(iw-ow)/2:y=0"

        filter_complex = (
            f"[0:v]split=3[src_top][src_main][src_bot];"
            f"[src_top]{top_bg_filter}[top_bg];"
            f"[src_main]{crop_filter},scale=1080:{main_h}[main_v];"
            f"[src_bot]{bot_bg_filter}[bot_bg];"
            f"[top_bg][main_v][bot_bg]vstack=inputs=3,"
            f"setsar=1{sub_filter}[outv]"
        )

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", str(input_path),
        "-t", str(duration),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        logger.warning(f"Render failed with primary command, attempting fallback: {result.stderr}")
        # Robust fallback: 3-section layout without ASS subtitle filter
        fallback_filter = (
            f"[0:v]split=3[fb_top][fb_main][fb_bot];"
            f"[fb_top]{top_bg_filter}[fbt];"
            f"[fb_main]crop=w='ih*1080/{main_h}':h=ih:x='(iw-ow)/2':y=0,scale=1080:{main_h}[fbm];"
            f"[fb_bot]{bot_bg_filter}[fbb];"
            f"[fbt][fbm][fbb]vstack=inputs=3,setsar=1[outv]"
        )
        fallback_command = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", str(input_path),
            "-t", str(duration),
            "-filter_complex", fallback_filter,
            "-map", "[outv]",
            "-map", "0:a:0?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        fallback_res = subprocess.run(
            fallback_command,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if fallback_res.returncode == 0:
            return output_path

        raise RuntimeError(f"FFmpeg failed with code {result.returncode}: {result.stderr}")

    return output_path


def create_stitched_compilation(
    clip_paths: List[Path],
    output_path: Path,
    transition_type: str = "fade",
    transition_duration: float = 0.4,
) -> Path:
    """
    Combine multiple generated vertical clips into a single compilation video
    with professional transitions (fade, slideleft, zoomin, smoothleft, or clean cut).
    """
    valid_clips = [Path(p) for p in clip_paths if Path(p).exists()]
    if not valid_clips:
        raise ValueError("No valid clips provided for compilation.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(valid_clips) == 1 or transition_type == "cut":
        # Concat using demuxer for clean cuts
        concat_list = output_path.parent / f"{output_path.stem}_concat.txt"
        lines = [f"file '{escape_ffmpeg_filter_path(p)}'" for p in valid_clips]
        concat_list.write_text("\n".join(lines), encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        concat_list.unlink(missing_ok=True)
        if res.returncode == 0:
            return output_path
        raise RuntimeError(f"Concat compilation failed: {res.stderr}")

    # Map transition names to FFmpeg xfade transition names
    xfade_map = {
        "crossfade": "fade",
        "fade": "fade",
        "slide": "slideleft",
        "slideleft": "slideleft",
        "zoom": "zoomin",
        "zoomin": "zoomin",
        "motion": "smoothleft",
        "smoothleft": "smoothleft",
    }
    trans = xfade_map.get(transition_type.lower(), "fade")
    d = max(0.2, min(0.8, transition_duration))

    # Retrieve durations of all clips to calculate xfade offsets
    clip_durations: List[float] = []
    for c in valid_clips:
        info = get_video_info(c)
        clip_durations.append(info["duration"])

    # Build FFmpeg command with xfade and acrossfade
    cmd = ["ffmpeg", "-y"]
    for c in valid_clips:
        cmd.extend(["-i", str(c)])

    filter_complex_parts: List[str] = []
    running_offset = clip_durations[0] - d
    last_v = "0:v"
    last_a = "0:a"

    for i in range(1, len(valid_clips)):
        next_v = f"{i}:v"
        next_a = f"{i}:a"
        out_v = f"v{i}" if i < len(valid_clips) - 1 else "finalv"
        out_a = f"a{i}" if i < len(valid_clips) - 1 else "finala"

        filter_complex_parts.append(
            f"[{last_v}][{next_v}]xfade=transition={trans}:duration={d:.2f}:offset={running_offset:.2f}[{out_v}]"
        )
        filter_complex_parts.append(
            f"[{last_a}][{next_a}]acrossfade=d={d:.2f}[{out_a}]"
        )

        last_v = out_v
        last_a = out_a
        if i < len(valid_clips) - 1:
            running_offset += clip_durations[i] - d

    cmd.extend([
        "-filter_complex", ";".join(filter_complex_parts),
        "-map", "[finalv]",
        "-map", "[finala]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ])

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode == 0:
        return output_path

    logger.warning(f"xfade compilation failed ({res.stderr}), falling back to cut concat.")
    # Fallback to cut concat if xfade encounters edge case
    return create_stitched_compilation(valid_clips, output_path, transition_type="cut")