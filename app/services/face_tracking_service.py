import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("clipforge.facetracking")


class FaceTrackingService:
    def __init__(self):
        # Load frontal and profile face cascades
        frontal_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
        if not frontal_path.exists():
            frontal_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        profile_path = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"

        self.frontal_cascade = cv2.CascadeClassifier(str(frontal_path))
        self.profile_cascade = cv2.CascadeClassifier(str(profile_path))

    def detect_faces_in_frame(self, frame_gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a grayscale image frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        # 1. Detect frontal faces
        frontal_faces = self.frontal_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(36, 36),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        detected = list(frontal_faces) if len(frontal_faces) > 0 else []

        # 2. If few or no frontal faces found, check profile faces (for side profiles)
        if len(detected) < 2:
            profiles = self.profile_cascade.detectMultiScale(
                frame_gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(36, 36),
            )
            if len(profiles) > 0:
                for px, py, pw, ph in profiles:
                    # Check overlap with existing detected faces
                    overlap = False
                    for fx, fy, fw, fh in detected:
                        if (
                            abs(px - fx) < max(pw, fw) * 0.6
                            and abs(py - fy) < max(ph, fh) * 0.6
                        ):
                            overlap = True
                            break
                    if not overlap:
                        detected.append((px, py, pw, ph))

        # Sort faces from largest (most prominent) to smallest
        detected.sort(key=lambda b: b[2] * b[3], reverse=True)
        return detected

    def analyze_clip(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        sample_fps: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Analyze face positions and dynamics within a clip segment [start_time, end_time].
        Determines:
        - Layout mode: 'single' or 'two_view'
        - Smooth camera crop trajectories
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video {video_path} for face tracking.")
            return self._build_fallback_result(1920, 1080)

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

        duration = max(0.5, end_time - start_time)
        step_seconds = 1.0 / max(1.0, sample_fps)
        sample_times = [
            round(start_time + i * step_seconds, 3)
            for i in range(int(math.ceil(duration / step_seconds)))
            if start_time + i * step_seconds <= end_time
        ]

        if not sample_times:
            sample_times = [start_time]

        # Calculate standard 9:16 vertical crop width for this video
        crop_w = int(round(video_h * 9.0 / 16.0))
        # Ensure crop width is an even number
        if crop_w % 2 != 0:
            crop_w += 1
        crop_w = min(crop_w, video_w)

        # Collect detections across sampled timestamps
        frames_data: List[Dict[str, Any]] = []

        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Downscale frame for fast detection if video is 1080p or larger
            scale = 1.0
            if video_w > 960:
                scale = 960.0 / video_w
                detect_frame = cv2.resize(frame, (960, int(video_h * scale)))
            else:
                detect_frame = frame

            gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            raw_faces = self.detect_faces_in_frame(gray)

            # Map coordinates back to original frame dimensions
            mapped_faces = []
            for rx, ry, rw, rh in raw_faces:
                fx = int(rx / scale)
                fy = int(ry / scale)
                fw = int(rw / scale)
                fh = int(rh / scale)
                cx = fx + fw / 2.0
                cy = fy + fh / 2.0
                mapped_faces.append({
                    "x": fx,
                    "y": fy,
                    "w": fw,
                    "h": fh,
                    "cx": cx,
                    "cy": cy,
                    "rel_cx": cx / video_w,
                    "rel_cy": cy / video_h,
                })

            frames_data.append({
                "time": t,
                "rel_time": t - start_time,
                "faces": mapped_faces,
            })

        cap.release()

        if not frames_data:
            return self._build_fallback_result(video_w, video_h)

        # Decide layout and compute smooth paths
        return self._process_detections(frames_data, video_w, video_h, crop_w, duration)

    def _process_detections(
        self,
        frames_data: List[Dict[str, Any]],
        video_w: int,
        video_h: int,
        crop_w: int,
        duration: float,
    ) -> Dict[str, Any]:
        """
        Process detections across frames to:
        1. Determine if video qualifies for Two-View (stacked) or Single-Subject layout.
        2. Generate deadband-filtered, temporally smoothed camera trajectories.
        """
        # Analyze two-subject separation
        two_subject_frames = 0
        left_positions: List[float] = []
        right_positions: List[float] = []
        all_primary_cxs: List[float] = []

        center_crop_x = (video_w - crop_w) / 2.0

        for fd in frames_data:
            faces = fd["faces"]
            if len(faces) >= 2:
                # Find the two largest faces
                f1, f2 = faces[0], faces[1]
                # Order by x position (left vs right)
                if f1["cx"] > f2["cx"]:
                    f1, f2 = f2, f1

                separation = f2["cx"] - f1["cx"]
                # If separation is larger than 70% of 9:16 crop width, both cannot fit in one vertical crop
                if separation > crop_w * 0.70:
                    two_subject_frames += 1
                    left_positions.append(f1["cx"])
                    right_positions.append(f2["cx"])

            if len(faces) > 0:
                all_primary_cxs.append(faces[0]["cx"])

        total_frames = len(frames_data)
        two_view_ratio = two_subject_frames / max(1, total_frames)

        # Select layout mode:
        # Require sustained dual presence (at least 25% of sampled frames have wide separated subjects)
        # and aspect ratio must be widescreen (16:9 or wider)
        is_widescreen = (video_w / video_h) >= 1.5
        use_two_view = is_widescreen and two_view_ratio >= 0.25 and len(left_positions) >= 3

        if use_two_view:
            # Mode B: Two-View Stacked Layout
            return self._build_two_view_trajectory(
                frames_data,
                left_positions,
                right_positions,
                video_w,
                video_h,
                crop_w,
                duration,
            )
        else:
            # Mode A: Single-Subject Tracking Layout
            return self._build_single_subject_trajectory(
                frames_data,
                all_primary_cxs,
                video_w,
                video_h,
                crop_w,
                duration,
            )

    def _build_single_subject_trajectory(
        self,
        frames_data: List[Dict[str, Any]],
        all_primary_cxs: List[float],
        video_w: int,
        video_h: int,
        crop_w: int,
        duration: float,
    ) -> Dict[str, Any]:
        """
        Compute smooth single-subject tracking trajectory with deadband and EMA smoothing.
        """
        default_x = (video_w - crop_w) / 2.0
        max_x = max(0, video_w - crop_w)

        # Initial target position: median of detected faces or center
        if all_primary_cxs:
            initial_cx = float(np.median(all_primary_cxs))
        else:
            initial_cx = video_w / 2.0

        current_x = np.clip(initial_cx - crop_w / 2.0, 0, max_x)
        last_reliable_x = current_x

        deadband_px = video_w * 0.04  # 4% deadband threshold to prevent micro-jitter
        smoothing_alpha = 0.22  # Exponential smoothing factor for natural camera pan

        smoothed_points: List[Dict[str, float]] = []

        for fd in frames_data:
            t = fd["rel_time"]
            faces = fd["faces"]

            if faces:
                # Find face closest to current tracked center for identity continuity
                best_face = min(
                    faces,
                    key=lambda f: abs(f["cx"] - (current_x + crop_w / 2.0)),
                )
                target_x = np.clip(best_face["cx"] - crop_w / 2.0, 0, max_x)
                last_reliable_x = target_x
            else:
                # Face lost momentarily: hold last reliable position
                target_x = last_reliable_x

            # Apply deadband: only move if movement exceeds threshold
            if abs(target_x - current_x) > deadband_px:
                current_x = current_x + smoothing_alpha * (target_x - current_x)

            current_x = float(np.clip(current_x, 0, max_x))
            smoothed_points.append({
                "time": round(t, 3),
                "crop_x": round(current_x, 1),
            })

        # Calculate average / dominant crop_x
        avg_crop_x = float(np.mean([p["crop_x"] for p in smoothed_points]))
        avg_crop_x = round(float(np.clip(avg_crop_x, 0, max_x)), 1)

        # Check if camera movement is significant
        min_x = min(p["crop_x"] for p in smoothed_points)
        max_x_val = max(p["crop_x"] for p in smoothed_points)
        has_dynamic_motion = (max_x_val - min_x) > (video_w * 0.06)

        # Track lowest extent of primary subject face to avoid subtitle overlap
        max_bottom = 0.0
        for fd in frames_data:
            for f in fd.get("faces", []):
                bottom = f["y"] + f["h"]
                if bottom > max_bottom:
                    max_bottom = bottom
        rel_face_bottom = round(max_bottom / video_h, 3) if max_bottom > 0 else 0.5

        return {
            "layout_mode": "single",
            "video_w": video_w,
            "video_h": video_h,
            "crop_w": crop_w,
            "crop_h": video_h,
            "dominant_crop_x": avg_crop_x,
            "rel_face_bottom": rel_face_bottom,
            "has_dynamic_motion": has_dynamic_motion,
            "trajectory": smoothed_points,
            "description": "Single-subject framing tracking the primary speaker with stabilized smoothing.",
        }

    def _build_two_view_trajectory(
        self,
        frames_data: List[Dict[str, Any]],
        left_positions: List[float],
        right_positions: List[float],
        video_w: int,
        video_h: int,
        crop_w: int,
        duration: float,
    ) -> Dict[str, Any]:
        """
        Compute positions for upper (Subject 1) and lower (Subject 2) views.
        """
        max_x = max(0, video_w - crop_w)

        # Median anchor positions for Left and Right subjects
        left_cx = float(np.median(left_positions)) if left_positions else (video_w * 0.25)
        right_cx = float(np.median(right_positions)) if right_positions else (video_w * 0.75)

        crop_x_left = float(np.clip(left_cx - crop_w / 2.0, 0, max_x))
        crop_x_right = float(np.clip(right_cx - crop_w / 2.0, 0, max_x))

        return {
            "layout_mode": "two_view",
            "video_w": video_w,
            "video_h": video_h,
            "crop_w": crop_w,
            "crop_h": video_h,
            "rel_face_bottom": 0.5,
            "top_view": {
                "subject": "Subject 1 (Left)",
                "crop_x": round(crop_x_left, 1),
                "crop_y": 0,
                "crop_w": crop_w,
                "crop_h": video_h,
            },
            "bottom_view": {
                "subject": "Subject 2 (Right)",
                "crop_x": round(crop_x_right, 1),
                "crop_y": 0,
                "crop_w": crop_w,
                "crop_h": video_h,
            },
            "description": "Synchronized two-view stacked layout displaying both conversational subjects.",
        }

    def _build_fallback_result(self, video_w: int, video_h: int) -> Dict[str, Any]:
        crop_w = int(round(video_h * 9.0 / 16.0))
        if crop_w % 2 != 0:
            crop_w += 1
        crop_w = min(crop_w, video_w)
        center_x = round((video_w - crop_w) / 2.0, 1)

        return {
            "layout_mode": "single",
            "video_w": video_w,
            "video_h": video_h,
            "crop_w": crop_w,
            "crop_h": video_h,
            "dominant_crop_x": center_x,
            "rel_face_bottom": 0.5,
            "has_dynamic_motion": False,
            "trajectory": [{"time": 0.0, "crop_x": center_x}],
            "description": "Centered 9:16 vertical crop (fallback).",
        }


# Singleton instance
face_tracker = FaceTrackingService()
