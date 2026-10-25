import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm"""
    seconds = max(0.0, seconds)
    milliseconds = round(seconds * 1000)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    secs = milliseconds // 1000
    milliseconds %= 1000

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds into ASS timestamp: H:MM:SS.cc (centiseconds)"""
    seconds = max(0.0, seconds)
    centiseconds = int(round(seconds * 100))

    hours = centiseconds // 360_000
    centiseconds %= 360_000

    minutes = centiseconds // 6_000
    centiseconds %= 6_000

    secs = centiseconds // 100
    centiseconds %= 100

    return f"{hours:01}:{minutes:02}:{secs:02}.{centiseconds:02}"


def format_chunk_text(words: List[Dict[str, Any]], style: str) -> str:
    """Format and clean subtitle chunk text based on styling aesthetic."""
    raw = " ".join(item.get("word", "").strip() for item in words if item.get("word")).strip()
    if not raw:
        return ""

    if style in {"viral", "modern"}:
        return raw.upper()

    # Professional & Minimal: clean sentence casing with punctuation preserved
    if len(raw) > 1:
        return raw[0].upper() + raw[1:]
    return raw.upper()


def wrap_subtitle_lines(text: str, max_chars_per_line: int = 18) -> str:
    """
    Wrap subtitle text into at most 2 balanced, highly readable lines separated by \\N.
    Prevents text clipping, safe margin violations, and keeps lines balanced.
    """
    text = text.strip()
    if len(text) <= max_chars_per_line or "\n" in text or "\\N" in text:
        return text

    words = text.split()
    if len(words) <= 1:
        return text

    # Find the split point that divides the phrase most evenly
    mid = len(text) / 2.0
    best_split = 1
    best_diff = 999.0
    curr_len = 0

    for i in range(len(words) - 1):
        curr_len += len(words[i]) + (1 if i > 0 else 0)
        diff = abs(curr_len - mid)
        if diff < best_diff:
            best_diff = diff
            best_split = i + 1

    line1 = " ".join(words[:best_split])
    line2 = " ".join(words[best_split:])
    return f"{line1}\\N{line2}"


def highlight_accent_word(text: str, accent_color: str = "&H0000FFFF&") -> str:
    """
    Highlight an important/emphasis word in a phrase using an accent color,
    while keeping the remaining text pure white (&H00FFFFFF&).
    Prioritizes numbers, all-caps emphasis, and content words over stopwords.
    """
    stopwords = {
        "a", "an", "the", "to", "in", "on", "at", "of", "for", "is", "are",
        "was", "were", "and", "or", "it", "with", "by", "be", "this", "that",
        "from", "into", "we", "you", "they", "i", "my", "your",
    }
    lines = text.split(r"\N")
    candidates = []

    for l_idx, line in enumerate(lines):
        line_words = line.split()
        for w_idx, w in enumerate(line_words):
            clean = re.sub(r"[^\w]", "", w)
            if not clean:
                continue
            score = 0
            if any(char.isdigit() for char in clean):
                score += 15
            if len(clean) >= 4 and clean.lower() not in stopwords:
                score += len(clean)
            candidates.append((score, l_idx, w_idx, clean))

    if not candidates:
        return text

    best = max(candidates, key=lambda c: c[0])
    if best[0] <= 0:
        return text

    target_l, target_w = best[1], best[2]

    rebuilt_lines = []
    for l_idx, line in enumerate(lines):
        lw = line.split()
        out_w = []
        for w_idx, w in enumerate(lw):
            if l_idx == target_l and w_idx == target_w:
                out_w.append(f"{{\\c{accent_color}}}{w}{{\\c&H00FFFFFF&}}")
            else:
                out_w.append(w)
        rebuilt_lines.append(" ".join(out_w))

    return r"\N".join(rebuilt_lines)


def build_dialogue_override_tags(
    animation: str = "pop_fade",
    glow_radius: float = 2.2,
    font_size_override: Optional[int] = None,
) -> str:
    """
    Build ASS dialogue inline override tags for professional outer glow and subtle animation.
    Supports:
    - pop_fade: pop-in scale-up (92% -> 100% over 120ms) + smooth fade-in (100ms)
    - pop: pop-in scale-up only
    - fade: smooth fade-in only
    - none: static text
    """
    tags = []
    if glow_radius and glow_radius > 0:
        tags.append(f"\\blur{glow_radius:.1f}")

    if font_size_override:
        tags.append(f"\\fs{font_size_override}")

    if animation == "pop_fade":
        tags.append(r"\fad(100,80)\fscx92\fscy92\t(0,120,\fscx100\fscy100)")
    elif animation == "pop":
        tags.append(r"\fscx92\fscy92\t(0,120,\fscx100\fscy100)")
    elif animation == "fade":
        tags.append(r"\fad(100,80)")

    if tags:
        return "{" + "".join(tags) + "}"
    return ""


def chunk_segments_for_display(
    segments: List[Dict[str, Any]],
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
    max_words_per_line: int = 4,
    style: str = "viral",
) -> List[Dict[str, Any]]:
    """
    Intelligently split Whisper segments into natural, high-retention subtitle lines.
    Respects sentence boundaries, conversational pauses, and punctuation cadence.
    """
    display_chunks: List[Dict[str, Any]] = []

    # Choose optimal word grouping per style
    limit = 3 if style == "viral" else (4 if style == "modern" else 5)
    limit = min(limit, max_words_per_line)

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Ignore segments outside clip bounds
        if seg_end <= clip_start:
            continue
        if clip_end is not None and seg_start >= clip_end:
            continue

        words = seg.get("words", [])
        if words and len(words) > 0:
            current_words: List[Dict[str, Any]] = []

            for idx, w in enumerate(words):
                w_start = w["start"]
                w_end = w["end"]

                if w_end <= clip_start:
                    continue
                if clip_end is not None and w_start >= clip_end:
                    continue

                # Check pause between words (> 0.35s is a natural breath/pause)
                has_speech_pause = False
                if current_words:
                    prev_end = current_words[-1]["end"]
                    if (w_start - prev_end) >= 0.35:
                        has_speech_pause = True

                # Check if previous word closed a sentence (. ? !)
                prev_closed_sentence = False
                if current_words:
                    prev_word = current_words[-1]["word"].rstrip()
                    if prev_word and prev_word[-1] in {".", "!", "?"}:
                        prev_closed_sentence = True

                # Flush current chunk if sentence ended, pause detected, or word limit reached
                if current_words and (prev_closed_sentence or has_speech_pause or len(current_words) >= limit):
                    c_start = max(0.0, current_words[0]["start"] - clip_start)
                    c_end = current_words[-1]["end"] - clip_start
                    if clip_end is not None:
                        c_end = min(clip_end - clip_start, c_end)

                    # Ensure minimum readable duration of 0.35s
                    c_end = max(c_end, c_start + 0.35)
                    text = format_chunk_text(current_words, style)

                    if c_end > c_start and text:
                        display_chunks.append({
                            "start": c_start,
                            "end": c_end,
                            "text": text,
                        })
                    current_words = []

                current_words.append(w)

            # Flush remaining words in segment
            if current_words:
                c_start = max(0.0, current_words[0]["start"] - clip_start)
                c_end = current_words[-1]["end"] - clip_start
                if clip_end is not None:
                    c_end = min(clip_end - clip_start, c_end)

                c_end = max(c_end, c_start + 0.35)
                text = format_chunk_text(current_words, style)

                if c_end > c_start and text:
                    display_chunks.append({
                        "start": c_start,
                        "end": c_end,
                        "text": text,
                    })

        else:
            # Fallback: estimate timings by dividing segment text evenly
            raw_tokens = seg.get("text", "").strip().split()
            if not raw_tokens:
                continue

            rel_start = max(0.0, seg_start - clip_start)
            rel_end = seg_end - clip_start
            if clip_end is not None:
                rel_end = min(clip_end - clip_start, rel_end)

            seg_duration = max(0.2, rel_end - rel_start)
            num_chunks = math.ceil(len(raw_tokens) / limit)
            chunk_duration = seg_duration / max(1, num_chunks)

            for i in range(num_chunks):
                tokens = raw_tokens[i * limit : (i + 1) * limit]
                c_start = rel_start + (i * chunk_duration)
                c_end = min(rel_end, rel_start + ((i + 1) * chunk_duration))
                text = format_chunk_text([{"word": t} for t in tokens], style)

                if c_end > c_start and text:
                    display_chunks.append({
                        "start": c_start,
                        "end": c_end,
                        "text": text,
                    })

    # Prevent chunk overlaps: ensure chunk[i].end <= chunk[i+1].start
    for i in range(len(display_chunks) - 1):
        if display_chunks[i]["end"] > display_chunks[i + 1]["start"]:
            display_chunks[i]["end"] = max(display_chunks[i]["start"] + 0.1, display_chunks[i + 1]["start"])

    return display_chunks


def create_srt(
    segments: List[Dict[str, Any]],
    output_path: Path,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
    style: str = "viral",
) -> Path:
    """Generate a clean SRT subtitle file with normalized timings."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = chunk_segments_for_display(
        segments=segments,
        clip_start=clip_start,
        clip_end=clip_end,
        max_words_per_line=4 if style in {"viral", "modern"} else 5,
        style=style,
    )

    entries = []
    for idx, chunk in enumerate(chunks, 1):
        entries.append(
            f"{idx}\n"
            f"{format_srt_timestamp(chunk['start'])} --> {format_srt_timestamp(chunk['end'])}\n"
            f"{chunk['text']}\n"
        )

    output_path.write_text("\n".join(entries), encoding="utf-8")
    return output_path


def get_ass_style_definition(
    style: str,
    layout_mode: str = "single",
    margin_v: Optional[int] = None,
    font_size: int = 74,
) -> str:
    """
    Returns bold, high-impact ASS style definitions optimized for 1080x1920 vertical canvas.
    Colors in ASS are &HAABBGGRR in hex.

    Font and Size:
    - Bold, modern sans-serif typeface (Arial Bold / Montserrat style).
    - Base font size: 64-88px (default 74px at 1080x1920).
    - Crisp pure white primary text (&H00FFFFFF).

    Glow and Outline:
    - Dark outline (4.5px) for clear separation over bright and complex backgrounds.
    - Subtle drop shadow (2.5px) with soft transparency.
    - Soft outer glow effect applied via blur tag.

    Positions:
    - Single-subject: MarginV=320 (lower-middle portion of main video section, safe above bottom blur).
    - Two-view stacked: MarginV=930 (centered around the middle divider, avoiding both speakers' faces).
    """
    if margin_v is None:
        if layout_mode == "two_view":
            margin_v = 930
        else:
            margin_v = 320

    # Ensure font size is within the required 64-88px range
    font_size = max(64, min(88, font_size))

    if style == "viral":
        # Viral Shorts: Bold modern Arial, pure white primary text, deep charcoal outline, soft shadow
        return (
            f"Style: Default,Arial,{font_size},&H00FFFFFF,&H0000FFFF,&H00101010,&HA0000000,"
            f"-1,0,0,0,100,100,1.0,0,1,4.5,2.5,2,80,80,{margin_v},1"
        )
    elif style == "minimal":
        # Elegant clean typography, refined off-white, subtle soft border
        return (
            f"Style: Default,Arial,58,&H00F5F5F5,&H00FFFFFF,&H00181818,&H70000000,"
            f"-1,0,0,0,100,100,0.5,0,1,2.5,1.5,2,80,80,{margin_v},1"
        )
    elif style == "modern":
        # Studio Modern: Crisp white core with subtle electric tint in shadow
        return (
            f"Style: Default,Arial,{font_size},&H00FFFFFF,&H00FFFFFF,&H00050505,&H80FFE500,"
            f"-1,0,0,0,100,100,1.1,0,1,4.2,2.5,2,80,80,{margin_v},1"
        )
    else:  # professional
        # Executive Studio: Crisp pure white (#FFFFFF), refined dark stroke, subtle soft shadow
        return (
            f"Style: Default,Arial,{font_size},&H00FFFFFF,&H00FFFFFF,&H00121212,&HB0000000,"
            f"-1,0,0,0,100,100,0.8,0,1,3.5,2.2,2,80,80,{margin_v},1"
        )


def calculate_adaptive_margin_v(
    layout_mode: str = "single",
    tracking_info: Optional[Dict[str, Any]] = None,
    bottom_h: int = 192,
) -> int:
    """
    Calculate safe vertical margin from canvas bottom to ensure:
    1. Subtitles stay inside the lower-middle portion of the main video section.
    2. Subtitles never overlap the bottom blurred background (at least 60px above bottom_h).
    3. If the primary speaker's face is positioned low in the frame, automatically adjust upward.
    """
    if layout_mode == "two_view":
        return 930

    base_margin = 320

    # Ensure margin sits safely above the bottom blurred section
    min_safe_margin = bottom_h + 80
    margin_v = max(base_margin, min_safe_margin)

    # If face is low in frame, adjust upward to avoid overlapping face
    if tracking_info:
        rel_face_bottom = tracking_info.get("rel_face_bottom")
        if rel_face_bottom is not None and rel_face_bottom > 0.65:
            # Face extends low: bump MarginV up to give clear clearance
            margin_v = max(margin_v, 420)

    return margin_v


def create_ass(
    segments: List[Dict[str, Any]],
    output_path: Path,
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
    style: str = "viral",
    layout_mode: str = "single",
    animation: str = "pop_fade",
    accent_color: Optional[str] = None,
    tracking_info: Optional[Dict[str, Any]] = None,
    font_size: int = 74,
) -> Path:
    """
    Generate an enhanced ASS subtitle file formatted for 1080x1920 vertical video.
    Features:
    - Large, bold sans-serif typeface (Arial Bold)
    - Pure white primary text with soft outer glow effect
    - Restrained pop-in / scale-up & fade-in animations synchronized with dialogue
    - Word highlighting with configurable accent color (default yellow for viral/modern)
    - Dynamic line wrapping to max 2 lines without clipping
    - Safe vertical placement inside the main video section avoiding face overlap.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = chunk_segments_for_display(
        segments=segments,
        clip_start=clip_start,
        clip_end=clip_end,
        max_words_per_line=4 if style in {"viral", "modern"} else 5,
        style=style,
    )

    # Determine adaptive MarginV
    margin_v = calculate_adaptive_margin_v(layout_mode=layout_mode, tracking_info=tracking_info)

    # Accent color for word highlighting in viral/modern styles
    if accent_color is None:
        if style == "viral":
            accent_color = "&H0000FFFF&"  # Warm energetic yellow in ASS &HAABBGGRR
        elif style == "modern":
            accent_color = "&H00FFFF00&"  # Electric cyan in ASS &HAABBGGRR
        else:
            accent_color = None

    header = (
        "[Script Info]\n"
        "Title: ClipForge Studio Subtitles\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{get_ass_style_definition(style, layout_mode=layout_mode, margin_v=margin_v, font_size=font_size)}\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for chunk in chunks:
        start_fmt = format_ass_timestamp(chunk["start"])
        end_fmt = format_ass_timestamp(chunk["end"])

        # Format and wrap long lines into max 2 balanced lines
        raw_text = chunk["text"]
        wrapped = wrap_subtitle_lines(raw_text, max_chars_per_line=18)

        # Dynamic font size adjustment for longer phrases
        line_chars = max(len(line) for line in wrapped.split("\\N"))
        font_override = None
        if line_chars > 20:
            font_override = max(64, font_size - 8)
        elif line_chars > 16:
            font_override = max(66, font_size - 4)

        # Apply word highlighting if accent color is enabled
        if accent_color and style in {"viral", "modern"}:
            highlighted_text = highlight_accent_word(wrapped, accent_color=accent_color)
        else:
            highlighted_text = wrapped

        # Build glow and animation tags
        override_tags = build_dialogue_override_tags(
            animation=animation,
            glow_radius=2.2,
            font_size_override=font_override,
        )

        final_dialogue_text = f"{override_tags}{highlighted_text}"
        events.append(f"Dialogue: 0,{start_fmt},{end_fmt},Default,,0,0,0,,{final_dialogue_text}")

    full_ass = header + "\n".join(events) + "\n"
    output_path.write_text(full_ass, encoding="utf-8")
    return output_path
