from typing import Any, Dict, List, Optional
import math


def score_clip_candidate(clip: Dict[str, Any]) -> float:
    """
    Score a candidate clip based on duration, speech density, and text engagement.
    """
    duration = clip["end"] - clip["start"]
    if duration <= 0:
        return 0.0

    text = clip.get("text", "")
    words = text.split()
    word_count = len(words)

    # Speech pace: ideal short-form speech is ~2 to 3.5 words/second
    words_per_sec = word_count / duration
    pace_score = min(1.0, words_per_sec / 2.5)

    # Ideal duration bonus (between 25 and 45 seconds is sweet spot for shorts)
    if 25.0 <= duration <= 45.0:
        duration_score = 1.0
    elif 15.0 <= duration < 25.0:
        duration_score = 0.8
    elif 45.0 < duration <= 60.0:
        duration_score = 0.85
    else:
        duration_score = 0.5

    # Engagement bonus: presence of questions, exclamation, numbers, high-energy terms
    engagement_bonus = 0.0
    if "?" in text:
        engagement_bonus += 0.15
    if "!" in text:
        engagement_bonus += 0.1
    if any(char.isdigit() for char in text):
        engagement_bonus += 0.1

    final_score = (pace_score * 0.45) + (duration_score * 0.35) + (engagement_bonus * 0.2)
    return round(final_score, 3)


def generate_candidate_clips(
    segments: List[Dict[str, Any]],
    total_duration: float,
    clip_count: int = 5,
    target_duration: int = 35,
    minimum_duration: int = 15,
    maximum_duration: int = 60,
) -> List[Dict[str, Any]]:
    """
    Intelligently select the best non-overlapping short-form clips
    matching the user's requested clip count.
    """
    # Fallback if no speech segments exist (e.g. silent video, B-roll, music)
    if not segments or len(segments) == 0:
        return generate_time_slice_clips(total_duration, clip_count, target_duration)

    # 1. Slide a window over transcript segments to build candidate pools
    raw_candidates: List[Dict[str, Any]] = []

    for i in range(len(segments)):
        seg_start = segments[i]["start"]
        clip_texts = []
        clip_segments = []

        for j in range(i, len(segments)):
            curr_seg = segments[j]
            curr_end = curr_seg["end"]
            curr_duration = curr_end - seg_start

            if curr_duration > maximum_duration:
                break

            clip_texts.append(curr_seg["text"])
            clip_segments.append(curr_seg)

            if curr_duration >= minimum_duration:
                raw_candidates.append({
                    "start": round(seg_start, 2),
                    "end": round(curr_end, 2),
                    "duration": round(curr_duration, 2),
                    "text": " ".join(clip_texts).strip(),
                    "segments": list(clip_segments),
                })

    # If no candidate satisfied the duration window (e.g. short video)
    if not raw_candidates:
        if total_duration > 5.0:
            return [{
                "start": 0.0,
                "end": round(total_duration, 2),
                "duration": round(total_duration, 2),
                "text": " ".join(s["text"] for s in segments).strip() or "Full Clip",
                "segments": segments,
                "score": 1.0,
            }]
        return []

    # 2. Score all candidates
    for cand in raw_candidates:
        cand["score"] = score_clip_candidate(cand)

    # Sort primarily by score descending
    raw_candidates.sort(key=lambda x: x["score"], reverse=True)

    # 3. Select top N non-overlapping candidates
    selected_clips: List[Dict[str, Any]] = []

    for cand in raw_candidates:
        # Check overlap with already selected clips
        overlaps = False
        for sel in selected_clips:
            # Overlap threshold: allow max 5s overlap
            overlap_start = max(cand["start"], sel["start"])
            overlap_end = min(cand["end"], sel["end"])
            if overlap_end - overlap_start > 5.0:
                overlaps = True
                break

        if not overlaps:
            selected_clips.append(cand)
            if len(selected_clips) >= clip_count:
                break

    # 4. If we couldn't get enough non-overlapping clips, fill with time-sliced partitions
    if len(selected_clips) < clip_count and total_duration >= minimum_duration * 1.5:
        remaining_needed = clip_count - len(selected_clips)
        time_clips = generate_time_slice_clips(
            total_duration,
            clip_count=remaining_needed,
            target_duration=target_duration,
            existing_clips=selected_clips,
        )
        for tc in time_clips:
            # Associate overlapping transcript text if any
            matched_text = [
                s["text"] for s in segments
                if not (s["end"] <= tc["start"] or s["start"] >= tc["end"])
            ]
            tc["text"] = " ".join(matched_text).strip() or f"Highlight Moment ({tc['start']}s - {tc['end']}s)"
            tc["segments"] = [
                s for s in segments
                if not (s["end"] <= tc["start"] or s["start"] >= tc["end"])
            ]
            selected_clips.append(tc)
            if len(selected_clips) >= clip_count:
                break

    # Sort final clips by start time for natural progression
    selected_clips.sort(key=lambda x: x["start"])
    return selected_clips[:clip_count]


def generate_time_slice_clips(
    total_duration: float,
    clip_count: int,
    target_duration: int = 35,
    existing_clips: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Generate time-distributed clips when transcript text is unavailable or insufficient."""
    if total_duration <= 0:
        return []

    existing = existing_clips or []
    clip_duration = min(float(target_duration), total_duration)

    if total_duration <= clip_duration:
        return [{
            "start": 0.0,
            "end": round(total_duration, 2),
            "duration": round(total_duration, 2),
            "text": "Full Highlight",
            "segments": [],
            "score": 0.7,
        }]

    clips: List[Dict[str, Any]] = []
    # Divide the timeline into evenly spaced intervals
    step = (total_duration - clip_duration) / max(1, clip_count)

    for i in range(clip_count):
        start = round(i * step, 2)
        end = round(min(total_duration, start + clip_duration), 2)
        duration = round(end - start, 2)

        # Check existing overlap
        overlaps = False
        for ex in existing:
            if max(start, ex["start"]) < min(end, ex["end"]):
                overlaps = True
                break

        if not overlaps and duration >= 5.0:
            clips.append({
                "start": start,
                "end": end,
                "duration": duration,
                "text": f"Clip {len(existing) + len(clips) + 1}",
                "segments": [],
                "score": 0.7,
            })

    return clips