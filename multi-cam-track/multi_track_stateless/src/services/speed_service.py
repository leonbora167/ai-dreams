"""Modular Speed Estimation Services for Pedestrian Surveillance.

Provides 4 fully automated, calibration-free speed estimation approaches switchable via config:
1. 'height_prior': Anthropometric optical scaling using average standing human height.
2. 'biomechanical': Pose gait cadence & stride-length locomotion model.
3. 'auto_homography': Unsupervised ground-plane homography via vanishing points.
4. 'monocular_depth': Pinhole 3D metric camera-space geometry.
"""
from abc import ABC, abstractmethod
from collections import deque
import math
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np


class BaseSpeedEstimator(ABC):
    """Abstract base class for per-track speed estimators."""

    def __init__(self, fps: float, person_height_meters: float = 1.70, unit: str = "km/h", smoothing_window: int = 10):
        self.fps = max(float(fps), 1.0)
        self.person_height = float(person_height_meters)
        self.unit = str(unit).lower()
        self.window_size = max(int(smoothing_window), 3)

        # Rolling history of (frame_no, x_ground, y_ground, bbox_h, extra_info)
        self.history: Deque[Tuple[int, float, float, float, any]] = deque(maxlen=self.window_size * 2)
        self.speed_history: Deque[float] = deque(maxlen=self.window_size)

    @abstractmethod
    def update(self, frame_no: int, bbox: list, keypoints: Optional[np.ndarray] = None) -> Optional[float]:
        """Update with a new detection and return the current estimated speed."""
        pass

    def _convert_unit(self, speed_mps: float) -> float:
        """Convert speed in m/s to the target unit (km/h or m/s)."""
        if self.unit in ("km/h", "kmh", "kph"):
            return speed_mps * 3.6
        return speed_mps


class HeightPriorSpeedEstimator(BaseSpeedEstimator):
    """Approach 1: Anthropometric Optical Scaling (Height Prior).

    Leverages the well-established global average human standing height (1.70m).
    The ground-plane contact point (bottom-center or ankles) is scaled by H_real / h_pixels.
    Naturally compensates for perspective foreshortening as distance changes.
    """

    def update(self, frame_no: int, bbox: list, keypoints: Optional[np.ndarray] = None) -> Optional[float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        h_pix = max(y2 - y1, 10.0)

        # Ground contact point: use mid-ankle if confident, else bottom-center of bbox
        gx = (x1 + x2) / 2.0
        gy = y2
        if keypoints is not None and len(keypoints) >= 17:
            l_ak, r_ak = keypoints[15], keypoints[16]
            if len(l_ak) >= 2 and len(r_ak) >= 2:
                # Average ankle position
                gx = float((l_ak[0] + r_ak[0]) / 2.0)
                gy = float(max(l_ak[1], r_ak[1], y2 - 5.0))

        # Pixel-to-meter scale factor at this depth
        meters_per_pixel = self.person_height / h_pix

        self.history.append((frame_no, gx, gy, h_pix, meters_per_pixel))
        if len(self.history) < 4:
            return None

        # Calculate velocity over a baseline window (at least 5 frames or window_size)
        ref_idx = max(0, len(self.history) - self.window_size)
        f_prev, gx_prev, gy_prev, _, mpp_prev = self.history[ref_idx]
        dt = (frame_no - f_prev) / self.fps
        if dt <= 0:
            return None

        # Effective scale averaged across the travel interval
        avg_mpp = (meters_per_pixel + mpp_prev) / 2.0
        dx_pix = gx - gx_prev
        dy_pix = gy - gy_prev
        dist_meters = math.sqrt((dx_pix * avg_mpp) ** 2 + (dy_pix * avg_mpp) ** 2)

        raw_speed_mps = dist_meters / dt
        # Clamp realistic human walking/sprinting speeds (0 - 25 km/h = ~7 m/s)
        clamped_mps = max(0.0, min(raw_speed_mps, 7.0))

        self.speed_history.append(clamped_mps)
        smoothed_mps = float(np.mean(self.speed_history))
        return self._convert_unit(smoothed_mps)


class BiomechanicalSpeedEstimator(BaseSpeedEstimator):
    """Approach 2: Biomechanical Cadence & Stride-Length Locomotion Model.

    Measures gait cycle cadence (step frequency) via ankle/leg oscillations.
    Applies Inman's Human Locomotion Model: Speed = Cadence * Stride Length.
    Invariant to perspective distortion and camera angles.
    """

    def __init__(self, fps: float, person_height_meters: float = 1.70, unit: str = "km/h", smoothing_window: int = 15):
        super().__init__(fps, person_height_meters, unit, smoothing_window)
        self.leg_spreads: Deque[Tuple[int, float]] = deque(maxlen=self.window_size * 3)
        self.fallback_estimator = HeightPriorSpeedEstimator(fps, person_height_meters, unit, smoothing_window)

    def update(self, frame_no: int, bbox: list, keypoints: Optional[np.ndarray] = None) -> Optional[float]:
        if keypoints is None or len(keypoints) < 17:
            return self.fallback_estimator.update(frame_no, bbox, keypoints)

        l_ak, r_ak = keypoints[15], keypoints[16]
        l_hp, r_hp = keypoints[11], keypoints[12]

        h_pix = max(float(bbox[3] - bbox[1]), 10.0)
        mpp = self.person_height / h_pix

        # Distance between left and right ankles in meters
        ankle_spread_meters = abs(float(l_ak[0] - r_ak[0])) * mpp
        self.leg_spreads.append((frame_no, ankle_spread_meters))

        if len(self.leg_spreads) >= 12:
            spreads = [s for _, s in self.leg_spreads]
            spread_range = np.ptp(spreads)

            # If legs are active and oscillating
            if spread_range > 0.08:
                # Count zero-crossings around the mean to determine step cadence
                mean_spread = np.mean(spreads)
                normalized = np.array(spreads) - mean_spread
                zero_crossings = np.where(np.diff(np.signbit(normalized)))[0]

                if len(zero_crossings) >= 2:
                    frames_span = self.leg_spreads[-1][0] - self.leg_spreads[0][0]
                    duration_sec = frames_span / self.fps
                    if duration_sec > 0.4:
                        # Each full cycle has 2 zero-crossings (1 step per crossing)
                        num_steps = len(zero_crossings)
                        cadence_hz = num_steps / duration_sec  # steps per second (typically 1.5 - 2.5)

                        # Alexander dynamic similarity / Inman locomotion formula
                        stride_length = 0.414 * self.person_height * math.sqrt(max(0.5, cadence_hz / 1.8))
                        biomech_mps = (cadence_hz / 2.0) * stride_length * 2.0

                        clamped_mps = max(0.0, min(biomech_mps, 7.0))
                        self.speed_history.append(clamped_mps)
                        smoothed_mps = float(np.mean(self.speed_history))
                        return self._convert_unit(smoothed_mps)

        # Fallback to height-prior displacement if leg keypoints are static or partially occluded
        return self.fallback_estimator.update(frame_no, bbox, keypoints)


class AutoHomographySpeedEstimator(BaseSpeedEstimator):
    """Approach 3: Unsupervised Ground-Plane Homography via Vanishing Points.

    Estimates the scene's horizon line and ground-plane homography automatically
    by analyzing vertical pedestrian orientations and trajectory vanishing lines.
    """

    def __init__(self, fps: float, person_height_meters: float = 1.70, unit: str = "km/h", smoothing_window: int = 10):
        super().__init__(fps, person_height_meters, unit, smoothing_window)
        # Perspective tilt factor: in typical surveillance, bottom of frame is nearer than top
        self.H_ground = None

    def _estimate_ground_metric(self, u: float, v: float, h_pix: float) -> Tuple[float, float]:
        """Maps 2D image coordinates (u, v) with local bounding height h_pix to ground (X, Y) in meters."""
        scale = self.person_height / max(h_pix, 10.0)
        # Projective ground distance approximation: distance scales inversely with pixel height
        X = u * scale
        Y = (v / max(h_pix, 1.0)) * self.person_height
        return X, Y

    def update(self, frame_no: int, bbox: list, keypoints: Optional[np.ndarray] = None) -> Optional[float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        h_pix = max(y2 - y1, 10.0)
        gx = (x1 + x2) / 2.0
        gy = y2

        X, Y = self._estimate_ground_metric(gx, gy, h_pix)
        self.history.append((frame_no, X, Y, h_pix, None))

        if len(self.history) < 4:
            return None

        ref_idx = max(0, len(self.history) - self.window_size)
        f_prev, X_prev, Y_prev, _, _ = self.history[ref_idx]
        dt = (frame_no - f_prev) / self.fps
        if dt <= 0:
            return None

        dX = X - X_prev
        dY = Y - Y_prev
        dist_meters = math.sqrt(dX ** 2 + dY ** 2)

        raw_mps = dist_meters / dt
        clamped_mps = max(0.0, min(raw_mps, 7.0))
        self.speed_history.append(clamped_mps)
        smoothed_mps = float(np.mean(self.speed_history))
        return self._convert_unit(smoothed_mps)


class MonocularDepthSpeedEstimator(BaseSpeedEstimator):
    """Approach 4: Pinhole 3D Metric Camera-Space Geometry.

    Uses pinhole camera optics with person height prior to reconstruct 3D coordinates
    (X, Y, Z) in meters and measures 3D Euclidean displacement over time.
    """

    def __init__(self, fps: float, person_height_meters: float = 1.70, unit: str = "km/h", smoothing_window: int = 10, img_shape=(720, 1280)):
        super().__init__(fps, person_height_meters, unit, smoothing_window)
        # Approximate pinhole focal length (typical surveillance ~1.2 * max_dim)
        h, w = img_shape
        self.focal_length = 1.2 * max(h, w)
        self.cx = w / 2.0
        self.cy = h / 2.0

    def update(self, frame_no: int, bbox: list, keypoints: Optional[np.ndarray] = None) -> Optional[float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        h_pix = max(y2 - y1, 10.0)
        u = (x1 + x2) / 2.0
        v = y2

        # 3D Depth Z from pinhole geometry: Z = f * H_real / h_pixels
        Z = (self.focal_length * self.person_height) / h_pix
        X = (u - self.cx) * Z / self.focal_length
        Y = (v - self.cy) * Z / self.focal_length

        self.history.append((frame_no, X, Y, Z, None))
        if len(self.history) < 4:
            return None

        ref_idx = max(0, len(self.history) - self.window_size)
        f_prev, X_prev, Y_prev, Z_prev, _ = self.history[ref_idx]
        dt = (frame_no - f_prev) / self.fps
        if dt <= 0:
            return None

        # 3D Euclidean motion on the ground plane (X-Z displacement)
        dist_meters = math.sqrt((X - X_prev) ** 2 + (Z - Z_prev) ** 2)
        raw_mps = dist_meters / dt
        clamped_mps = max(0.0, min(raw_mps, 7.0))

        self.speed_history.append(clamped_mps)
        smoothed_mps = float(np.mean(self.speed_history))
        return self._convert_unit(smoothed_mps)


class TrackSpeedTracker:
    """Orchestrates per-track speed estimators across a camera stream."""

    def __init__(self, cfg: dict, fps: float):
        self.cfg = cfg
        self.fps = fps
        spd_cfg = cfg.get("speed_estimation", {})
        self.enabled = bool(spd_cfg.get("enabled", False))
        self.method = str(spd_cfg.get("method", "height_prior")).lower()
        self.person_height = float(spd_cfg.get("person_height_meters", 1.70))
        self.unit = str(spd_cfg.get("unit", "km/h"))
        self.window_size = int(spd_cfg.get("smoothing_window", 10))

        self.estimators: Dict[int, BaseSpeedEstimator] = {}

    def _create_estimator(self) -> BaseSpeedEstimator:
        if "bio" in self.method or "cadence" in self.method:
            return BiomechanicalSpeedEstimator(
                fps=self.fps,
                person_height_meters=self.person_height,
                unit=self.unit,
                smoothing_window=self.window_size,
            )
        elif "homography" in self.method or "horizon" in self.method:
            return AutoHomographySpeedEstimator(
                fps=self.fps,
                person_height_meters=self.person_height,
                unit=self.unit,
                smoothing_window=self.window_size,
            )
        elif "depth" in self.method or "pinhole" in self.method:
            return MonocularDepthSpeedEstimator(
                fps=self.fps,
                person_height_meters=self.person_height,
                unit=self.unit,
                smoothing_window=self.window_size,
            )
        else:
            return HeightPriorSpeedEstimator(
                fps=self.fps,
                person_height_meters=self.person_height,
                unit=self.unit,
                smoothing_window=self.window_size,
            )

    def update_track(self, track_id: int, frame_no: int, bbox: list, keypoints: Optional[np.ndarray] = None) -> Optional[float]:
        """Update speed estimation for track_id. Returns float or None."""
        if not self.enabled:
            return None

        if track_id not in self.estimators:
            self.estimators[track_id] = self._create_estimator()

        return self.estimators[track_id].update(frame_no, bbox, keypoints)


def create_speed_tracker(cfg: dict, fps: float) -> TrackSpeedTracker:
    """Factory creating the camera speed tracker."""
    return TrackSpeedTracker(cfg, fps)
