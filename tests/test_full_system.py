import asyncio
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from fastapi import BackgroundTasks, HTTPException
from app.main import (
    ProcessUrlRequest,
    UpdateTitleRequest,
    ValidateUrlRequest,
    health,
    process_youtube_url,
    update_title,
    validate_url,
)
from app.services.clipping_service import (
    generate_candidate_clips,
    generate_time_slice_clips,
    score_clip_candidate,
)
from app.services.face_tracking_service import FaceTrackingService, face_tracker
from app.services.pipeline_service import (
    create_job,
    get_job_state,
    process_video_job,
    save_job_state,
)
from app.services.subtitle_service import (
    build_dialogue_override_tags,
    calculate_adaptive_margin_v,
    create_ass,
    create_srt,
    format_ass_timestamp,
    format_srt_timestamp,
    highlight_accent_word,
    wrap_subtitle_lines,
)
from app.services.title_service import (
    ensure_mrbeast_title,
    generate_clip_title,
    parse_srt_text,
    validate_mrbeast_title,
)
from app.services.video_service import (
    build_smooth_crop_expr,
    calculate_layout_dimensions,
    create_stitched_compilation,
    create_vertical_clip,
    extract_thumbnail,
    get_video_info,
)
from app.services.youtube_service import (
    clean_yt_error_message,
    extract_youtube_info,
    get_js_runtimes_config,
    is_valid_youtube_url,
)


class TestClipForgeSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = PROJECT_DIR / "storage" / "temp"
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        cls.test_video = cls.temp_dir / "test_synth.mp4"

        # Generate a synthetic 6-second test MP4 (1280x720) with audio tone using ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=6:size=1280x720:rate=30",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=6",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            str(cls.test_video),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Could not generate test video: {res.stderr}")

    def test_01_timestamp_formatters(self):
        srt_time = format_srt_timestamp(65.123)
        self.assertEqual(srt_time, "00:01:05,123")

        ass_time = format_ass_timestamp(65.123)
        self.assertEqual(ass_time, "0:01:05.12")

    def test_02_subtitle_generation_single_and_two_view(self):
        sample_segments = [
            {
                "start": 0.5,
                "end": 2.5,
                "text": "Welcome to ClipForge AI video studio",
                "words": [
                    {"word": "Welcome", "start": 0.5, "end": 0.9},
                    {"word": "to", "start": 0.9, "end": 1.1},
                    {"word": "ClipForge", "start": 1.1, "end": 1.7},
                    {"word": "AI", "start": 1.7, "end": 2.0},
                    {"word": "video", "start": 2.0, "end": 2.3},
                    {"word": "studio", "start": 2.3, "end": 2.5},
                ],
            },
            {
                "start": 2.8,
                "end": 5.0,
                "text": "Transform horizontal content into viral shorts",
                "words": [],
            },
        ]

        srt_path = self.temp_dir / "test_sub.srt"
        ass_path_single = self.temp_dir / "test_sub_single.ass"
        ass_path_two_view = self.temp_dir / "test_sub_two_view.ass"

        create_srt(sample_segments, srt_path, clip_start=0.0, clip_end=6.0, style="viral")
        self.assertTrue(srt_path.exists())
        srt_content = srt_path.read_text(encoding="utf-8")
        self.assertIn("-->", srt_content)
        self.assertIn("CLIPFORGE", srt_content)

        # Single layout ASS
        create_ass(sample_segments, ass_path_single, clip_start=0.0, clip_end=6.0, style="viral", layout_mode="single")
        self.assertTrue(ass_path_single.exists())
        single_content = ass_path_single.read_text(encoding="utf-8")
        self.assertIn("320,1", single_content)  # MarginV=320

        # Two-view layout ASS (centered around divider)
        create_ass(sample_segments, ass_path_two_view, clip_start=0.0, clip_end=6.0, style="viral", layout_mode="two_view")
        self.assertTrue(ass_path_two_view.exists())
        two_view_content = ass_path_two_view.read_text(encoding="utf-8")
        self.assertIn("930,1", two_view_content)  # MarginV=930

    def test_03_clipping_logic(self):
        segments = [
            {"start": 0.0, "end": 8.0, "text": "This is segment one with interesting ideas!"},
            {"start": 8.5, "end": 16.0, "text": "Did you know how fast AI can create shorts?"},
            {"start": 16.5, "end": 26.0, "text": "Here is the key takeaway for video creators."},
            {"start": 26.5, "end": 35.0, "text": "Summary of everything we covered today!"},
        ]

        clips = generate_candidate_clips(
            segments=segments,
            total_duration=36.0,
            clip_count=3,
            target_duration=15,
            minimum_duration=8,
            maximum_duration=25,
        )

        self.assertGreaterEqual(len(clips), 1)
        self.assertLessEqual(len(clips), 3)
        for clip in clips:
            self.assertIn("start", clip)
            self.assertIn("end", clip)
            self.assertGreater(clip["end"], clip["start"])

        # Test silent video time-slice fallback
        silent_clips = generate_time_slice_clips(total_duration=60.0, clip_count=3, target_duration=15)
        self.assertEqual(len(silent_clips), 3)

    def test_04_youtube_url_validation(self):
        # Valid URLs
        self.assertTrue(is_valid_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(is_valid_youtube_url("https://youtu.be/dQw4w9WgXcQ"))
        self.assertTrue(is_valid_youtube_url("https://youtube.com/shorts/dQw4w9WgXcQ"))
        self.assertTrue(is_valid_youtube_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(is_valid_youtube_url("http://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share"))

        # Invalid URLs
        self.assertFalse(is_valid_youtube_url("https://vimeo.com/1234567"))
        self.assertFalse(is_valid_youtube_url("https://tiktok.com/@user/video/123"))
        self.assertFalse(is_valid_youtube_url("not a url"))
        self.assertFalse(is_valid_youtube_url(""))

    def test_05_face_tracking_and_layout_selection(self):
        # 1. Test fallback on synthetic video
        result = face_tracker.analyze_clip(self.test_video, start_time=0.0, end_time=2.0)
        self.assertIn("layout_mode", result)
        self.assertEqual(result["layout_mode"], "single")
        self.assertEqual(result["crop_w"] % 2, 0)
        self.assertAlmostEqual(result["crop_w"], 720 * 9 / 16, delta=2)

        # 2. Test two-view decision logic on simulated dual-face detections
        # Simulate video width 1920, height 1080, crop_w 608
        simulated_frames = [
            {
                "time": float(i),
                "rel_time": float(i),
                "faces": [
                    {"x": 200, "y": 200, "w": 100, "h": 100, "cx": 250.0, "cy": 250.0},
                    {"x": 1500, "y": 200, "w": 100, "h": 100, "cx": 1550.0, "cy": 250.0},
                ],
            }
            for i in range(10)
        ]
        decision = face_tracker._process_detections(
            simulated_frames, video_w=1920, video_h=1080, crop_w=608, duration=10.0
        )
        self.assertEqual(decision["layout_mode"], "two_view")
        self.assertIn("top_view", decision)
        self.assertIn("bottom_view", decision)
        self.assertLess(decision["top_view"]["crop_x"], decision["bottom_view"]["crop_x"])

    def test_06_video_service_single_and_two_view_rendering(self):
        info = get_video_info(self.test_video)
        self.assertAlmostEqual(info["duration"], 6.0, delta=0.5)

        # 1. Mode A: Single-subject vertical clip with smooth crop trajectory
        out_single = self.temp_dir / "test_out_single.mp4"
        ass_sub = self.temp_dir / "test_sub_single.ass"

        tracking_single = {
            "layout_mode": "single",
            "video_w": 1280,
            "video_h": 720,
            "crop_w": 405,
            "crop_h": 720,
            "dominant_crop_x": 437.0,
            "has_dynamic_motion": True,
            "trajectory": [
                {"time": 0.0, "crop_x": 400.0},
                {"time": 1.5, "crop_x": 450.0},
                {"time": 3.0, "crop_x": 420.0},
            ],
        }

        create_vertical_clip(
            input_path=self.test_video,
            output_path=out_single,
            start=0.5,
            end=3.5,
            subtitle_path=ass_sub,
            tracking_info=tracking_single,
        )

        self.assertTrue(out_single.exists())
        info_single = get_video_info(out_single)
        self.assertEqual(info_single["width"], 1080)
        self.assertEqual(info_single["height"], 1920)

        # 2. Mode B: Stacked two-view vertical clip
        out_two_view = self.temp_dir / "test_out_two_view.mp4"
        ass_sub_tv = self.temp_dir / "test_sub_two_view.ass"

        tracking_two_view = {
            "layout_mode": "two_view",
            "video_w": 1280,
            "video_h": 720,
            "crop_w": 405,
            "crop_h": 720,
            "top_view": {"crop_x": 100, "crop_y": 0, "crop_w": 405, "crop_h": 720},
            "bottom_view": {"crop_x": 800, "crop_y": 0, "crop_w": 405, "crop_h": 720},
        }

        create_vertical_clip(
            input_path=self.test_video,
            output_path=out_two_view,
            start=0.5,
            end=3.5,
            subtitle_path=ass_sub_tv,
            tracking_info=tracking_two_view,
        )

        self.assertTrue(out_two_view.exists())
        info_two_view = get_video_info(out_two_view)
        self.assertEqual(info_two_view["width"], 1080)
        self.assertEqual(info_two_view["height"], 1920)

    def test_07_compilation_with_transitions(self):
        clip1 = self.temp_dir / "test_out_single.mp4"
        clip2 = self.temp_dir / "test_out_two_view.mp4"

        comp_path = self.temp_dir / "test_out_compilation.mp4"
        create_stitched_compilation(
            clip_paths=[clip1, clip2],
            output_path=comp_path,
            transition_type="fade",
            transition_duration=0.3,
        )

        self.assertTrue(comp_path.exists())
        comp_info = get_video_info(comp_path)
        self.assertEqual(comp_info["width"], 1080)
        self.assertEqual(comp_info["height"], 1920)
        self.assertTrue(comp_info["has_audio"])

    def test_08_pipeline_job_execution_with_local_file(self):
        upload_dir = PROJECT_DIR / "storage" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_name = "test_pipeline_run.mp4"
        dest_video = upload_dir / upload_name
        dest_video.write_bytes(self.test_video.read_bytes())

        job_id = create_job(
            video_filename=upload_name,
            clip_count=3,
            subtitle_style="viral",
            transition_style="fade",
        )
        self.assertIsNotNone(job_id)

        # Run pipeline synchronously for test
        process_video_job(job_id)

        job_state = get_job_state(job_id)
        self.assertEqual(job_state["status"], "completed")
        self.assertEqual(job_state["progress"], 100)
        self.assertGreater(len(job_state["clips"]), 0)

        first_clip = job_state["clips"][0]
        self.assertIn("layout_mode", first_clip)
        self.assertIn("video_url", first_clip)
        self.assertIn("title", first_clip)
        self.assertIn("MrBeast", first_clip["title"])
        clip_file = PROJECT_DIR / first_clip["video_url"].lstrip("/")
        self.assertTrue(clip_file.exists())

        # Verify compilation reel was created for multiple clips
        if len(job_state["clips"]) >= 2:
            self.assertIsNotNone(job_state.get("compilation"))

    def test_09_api_endpoints(self):
        # Health endpoint
        res = asyncio.run(health())
        self.assertEqual(res["status"], "ok")

        # Validate URL endpoint - invalid URL
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(validate_url(ValidateUrlRequest(url="not-valid")))
        self.assertEqual(ctx.exception.status_code, 400)

        # Validate URL endpoint - valid URL format with mock
        with patch("app.main.extract_youtube_info") as mock_extract:
            mock_extract.return_value = {
                "video_id": "test1234567",
                "title": "Sample Interview Video",
                "duration": 180.0,
                "channel": "Tech Podcast",
                "thumbnail": "https://img.youtube.com/vi/test1234567/0.jpg",
                "url": "https://www.youtube.com/watch?v=test1234567",
            }
            data = asyncio.run(validate_url(ValidateUrlRequest(url="https://www.youtube.com/watch?v=test1234567")))
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["title"], "Sample Interview Video")
            self.assertEqual(data["channel"], "Tech Podcast")

        # Process URL endpoint
        bg_tasks = BackgroundTasks()
        req = ProcessUrlRequest(
            url="https://www.youtube.com/watch?v=test1234567",
            clip_count=5,
            subtitle_style="viral",
            transition_style="fade",
        )
        job_data = asyncio.run(process_youtube_url(req, bg_tasks))
        self.assertEqual(job_data["status"], "queued")
        self.assertIn("job_id", job_data)

    def test_10_enhanced_layout_dimensions_and_rendering(self):
        # 1. Default dimensions: 10% top, 80% main, 10% bottom
        dims_default = calculate_layout_dimensions(1080, 1920, top_ratio=0.10, bottom_ratio=0.10)
        self.assertEqual(dims_default["top_h"], 192)
        self.assertEqual(dims_default["main_h"], 1536)
        self.assertEqual(dims_default["bottom_h"], 192)
        self.assertEqual(dims_default["top_h"] + dims_default["main_h"] + dims_default["bottom_h"], 1920)
        self.assertEqual(dims_default["main_y"], 192)

        # 2. Configurable dimensions: 8% top, 12% bottom -> 80% main
        dims_custom = calculate_layout_dimensions(1080, 1920, top_ratio=0.08, bottom_ratio=0.12)
        self.assertEqual(dims_custom["top_h"] + dims_custom["main_h"] + dims_custom["bottom_h"], 1920)
        self.assertTrue(1450 <= dims_custom["main_h"] <= 1580)
        self.assertTrue(dims_custom["top_h"] % 2 == 0)
        self.assertTrue(dims_custom["bottom_h"] % 2 == 0)
        self.assertTrue(dims_custom["main_h"] % 2 == 0)

        # 3. Test video rendering with configurable layout
        out_custom_layout = self.temp_dir / "test_out_custom_layout.mp4"
        tracking = {
            "layout_mode": "single",
            "video_w": 1280,
            "video_h": 720,
            "crop_w": 405,
            "crop_h": 720,
            "dominant_crop_x": 437.0,
        }
        create_vertical_clip(
            input_path=self.test_video,
            output_path=out_custom_layout,
            start=0.0,
            end=2.0,
            tracking_info=tracking,
            top_ratio=0.12,
            bottom_ratio=0.12,
        )
        self.assertTrue(out_custom_layout.exists())
        info = get_video_info(out_custom_layout)
        self.assertEqual(info["width"], 1080)
        self.assertEqual(info["height"], 1920)
        self.assertTrue(info["has_audio"])

    def test_11_enhanced_subtitle_glow_and_animation(self):
        # 1. Line wrapping logic (max 2 lines)
        short_text = "AI VIDEO STUDIO"
        self.assertEqual(wrap_subtitle_lines(short_text, max_chars_per_line=18), short_text)

        long_text = "TRANSFORM HORIZONTAL CONTENT INTO VIRAL SHORTS"
        wrapped = wrap_subtitle_lines(long_text, max_chars_per_line=18)
        self.assertIn(r"\N", wrapped)
        self.assertEqual(len(wrapped.split(r"\N")), 2)

        # 2. Accent word highlighting with yellow
        hl_text = highlight_accent_word("TRANSFORM HORIZONTAL CONTENT", accent_color="&H0000FFFF&")
        self.assertIn(r"{\c&H0000FFFF&}", hl_text)
        self.assertIn(r"{\c&H00FFFFFF&}", hl_text)

        # 3. Dialogue override tags: glow + pop-in animation
        override = build_dialogue_override_tags(animation="pop_fade", glow_radius=2.2, font_size_override=74)
        self.assertIn(r"\blur2.2", override)
        self.assertIn(r"\fad(100,80)", override)
        self.assertIn(r"\fscx92", override)
        self.assertIn(r"\fs74", override)

        # 4. Adaptive margin avoids face overlap
        normal_margin = calculate_adaptive_margin_v(layout_mode="single", tracking_info=None)
        self.assertEqual(normal_margin, 320)

        # If face is low in frame, margin is raised upward
        low_face_info = {"rel_face_bottom": 0.78}
        adjusted_margin = calculate_adaptive_margin_v(layout_mode="single", tracking_info=low_face_info)
        self.assertGreaterEqual(adjusted_margin, 420)

        # 5. ASS creation with glow and animation
        sample_segments = [
            {
                "start": 0.5,
                "end": 2.0,
                "text": "Produce professional vertical shorts with glowing captions",
                "words": [
                    {"word": "Produce", "start": 0.5, "end": 0.8},
                    {"word": "professional", "start": 0.8, "end": 1.2},
                    {"word": "vertical", "start": 1.2, "end": 1.6},
                    {"word": "shorts", "start": 1.6, "end": 2.0},
                ],
            }
        ]
        ass_path = self.temp_dir / "test_glow_enhanced.ass"
        create_ass(
            sample_segments,
            ass_path,
            clip_start=0.0,
            clip_end=3.0,
            style="viral",
            animation="pop_fade",
            font_size=76,
        )
        self.assertTrue(ass_path.exists())
        content = ass_path.read_text(encoding="utf-8")
        self.assertIn("PlayResX: 1080", content)
        self.assertIn("PlayResY: 1920", content)
        self.assertIn(r"\blur2.2", content)
        self.assertIn(r"\fad", content)
        self.assertIn("76", content)  # Fontsize 76 in style definition

    def test_12_automatic_mrbeast_title_generation(self):
        # 1. Test SRT dialogue parsing
        sample_srt = self.temp_dir / "test_title_sample.srt"
        sample_srt.write_text(
            "1\n00:00:00,500 --> 00:00:02,000\nWe built an entire city for 100 people\n\n"
            "2\n00:00:02,100 --> 00:00:04,500\nAnd gave away $1,000,000 to the final winner\n",
            encoding="utf-8",
        )
        parsed_text = parse_srt_text(sample_srt)
        self.assertIn("built an entire city", parsed_text)
        self.assertIn("$1,000,000", parsed_text)

        # 2. Test title generation from SRT: MUST include 'MrBeast'
        title_from_srt = generate_clip_title(srt_path=sample_srt, clip_index=1, total_clips=3)
        self.assertTrue(validate_mrbeast_title(title_from_srt))
        self.assertIn("MrBeast", title_from_srt)
        self.assertGreaterEqual(len(title_from_srt), 30)
        self.assertLessEqual(len(title_from_srt), 75)

        # 3. Test fallback to transcript text when SRT is missing
        missing_srt = self.temp_dir / "non_existent.srt"
        transcript_text = "I survived 24 hours trapped in the world's most extreme laser trap room"
        title_from_text = generate_clip_title(
            clip_text=transcript_text,
            srt_path=missing_srt,
            clip_index=2,
            total_clips=3,
        )
        self.assertTrue(validate_mrbeast_title(title_from_text))
        self.assertIn("MrBeast", title_from_text)

        # 4. Test fallback to video title when transcript is empty
        title_from_source = generate_clip_title(
            clip_text="",
            srt_path=None,
            source_title="Insane Hundred Hour Survival Challenge",
            clip_index=3,
            total_clips=3,
        )
        self.assertTrue(validate_mrbeast_title(title_from_source))
        self.assertIn("MrBeast", title_from_source)

        # 5. Test uniqueness across multiple clips from the same source video
        used_titles = set()
        clip_transcripts = [
            "We built an entire city in the desert for this challenge",
            "I gave away one million dollars to the last person standing",
            "Everything went completely wrong during the final sixty seconds",
            "We created an insane studio using AI tools and neural models",
            "The laser maze was impossible to escape before time ran out",
        ]
        generated_batch = []
        for i, text in enumerate(clip_transcripts, 1):
            t = generate_clip_title(
                clip_text=text,
                clip_index=i,
                total_clips=len(clip_transcripts),
                used_titles=used_titles,
            )
            generated_batch.append(t)
            self.assertIn("MrBeast", t)
            self.assertTrue(validate_mrbeast_title(t))

        # Check all titles are distinct (unique)
        self.assertEqual(len(generated_batch), len(set(generated_batch)))

        # 6. Test ensure_mrbeast_title corrector
        corrected = ensure_mrbeast_title("Gives Away $500,000 in Private Jet")
        self.assertIn("MrBeast", corrected)
        self.assertTrue(validate_mrbeast_title(corrected))

        # 7. Test API endpoint for updating clip title
        test_job_id = "test_job_title_update"
        save_job_state(
            test_job_id,
            {
                "job_id": test_job_id,
                "status": "completed",
                "clips": [
                    {
                        "clip_id": f"{test_job_id}_clip_1",
                        "title": "MrBeast Initial Title",
                    }
                ],
            },
        )
        req = UpdateTitleRequest(title="MrBeast's Updated Custom Challenge Title")
        update_res = asyncio.run(
            update_title(test_job_id, f"{test_job_id}_clip_1", req)
        )
        self.assertEqual(update_res["status"], "ok")

        updated_job = get_job_state(test_job_id)
        self.assertEqual(
            updated_job["clips"][0]["title"],
            "MrBeast's Updated Custom Challenge Title",
        )

    def test_13_youtube_error_cleaning_and_js_runtime(self):
        # 1. Verify ANSI escape code stripping does not produce the '[0' bug
        ansi_err = "\x1b[0;31mERROR:\x1b[0m [youtube] gTKS8SAwUzE: HTTP Error 403: Forbidden; please report this issue"
        cleaned = clean_yt_error_message(ansi_err)
        self.assertNotIn("\x1b", cleaned)
        self.assertNotEqual(cleaned, "[0")
        self.assertIn("HTTP Error 403", cleaned)

        # 2. Test bot / 403 detection friendly message
        self.assertTrue(cleaned.startswith("YouTube blocked or throttled the download"))

        # 3. Test private / sign-in error messages
        priv_err = clean_yt_error_message("ERROR: [youtube] 12345: Private video")
        self.assertEqual(priv_err, "This YouTube video is private and cannot be processed.")

        # 4. Test timeout error message
        timeout_err = clean_yt_error_message("\x1b[0;31mERROR:\x1b[0m [youtube] Connection timed out")
        self.assertIn("timed out while retrieving video", timeout_err)

        # 5. Verify JS runtimes auto-detection detects node when available
        runtimes = get_js_runtimes_config()
        self.assertIsInstance(runtimes, dict)


if __name__ == "__main__":
    unittest.main()
