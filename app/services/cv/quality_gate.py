"""Advanced Camera & Eye Tracking Quality Gate for RemiCare Strabismus AI.

Multi-factor quality assessment before running clinical screening:
- Face bounding box ratio & physical distance check (D_cm = 4095 / irisDistancePx)
- Head pose estimation (Yaw, Pitch, Roll angles)
- Eye Aspect Ratio (EAR) for blink exclusion
- Image blur / sharpness via Laplacian variance
- Illumination & lighting check (underexposure / overexposure / glare)
- Eye occlusion / visibility detection
- Minimum anatomical ocular resolution
"""

from dataclasses import dataclass, field
import math
from typing import Any, List, Optional, Tuple, Union
import numpy as np

try:
    # pyrefly: ignore [missing-import]
    import cv2
except ImportError:
    cv2 = None


@dataclass
class QualityGateResult:
    is_acceptable: bool
    camera_quality_score: float    # [0.0, 1.0]
    tracking_quality_score: float  # [0.0, 1.0]
    failure_reasons: List[str] = field(default_factory=list)
    user_guidance: Optional[str] = None
    # Detailed metrics
    blur_variance: Optional[float] = None
    brightness: Optional[float] = None
    estimated_distance_cm: Optional[float] = None


class AdvancedQualityGate:
    """Rigorous gate checking illumination, resolution, blur, occlusions, and head alignment."""

    def __init__(
        self,
        min_face_ratio: float = 0.20,
        max_face_ratio: float = 0.70,
        min_eye_width_px: float = 35.0,
        min_blur_var: float = 100.0,
        max_yaw_deg: float = 12.0,
        max_pitch_deg: float = 10.0,
        max_roll_deg: float = 6.0,
        min_ear_blink: float = 0.18,
        min_brightness: float = 40.0,
        max_brightness: float = 235.0,
        min_distance_cm: float = 35.0,
        max_distance_cm: float = 85.0,
    ):
        self.min_face_ratio = min_face_ratio
        self.max_face_ratio = max_face_ratio
        self.min_eye_width_px = min_eye_width_px
        self.min_blur_var = min_blur_var
        self.max_yaw_deg = max_yaw_deg
        self.max_pitch_deg = max_pitch_deg
        self.max_roll_deg = max_roll_deg
        self.min_ear_blink = min_ear_blink
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_distance_cm = min_distance_cm
        self.max_distance_cm = max_distance_cm

    @staticmethod
    def compute_ear(eye_landmarks: np.ndarray) -> float:
        """Computes Eye Aspect Ratio (EAR) from 6 2D eyelid points.
        
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
        """
        if eye_landmarks.shape[0] < 6:
            return 0.30
        p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
        d_v1 = float(np.linalg.norm(p2 - p6))
        d_v2 = float(np.linalg.norm(p3 - p5))
        d_h = float(np.linalg.norm(p1 - p4))
        if d_h < 1e-5:
            return 0.0
        return float((d_v1 + d_v2) / (2.0 * d_h))

    @staticmethod
    def compute_image_metrics(frame: np.ndarray) -> Tuple[float, float]:
        """Calculates Laplacian blur variance and mean pixel brightness."""
        if cv2 is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(np.mean(gray))
            return blur_var, brightness

        # NumPy fallback for blur variance (Laplacian kernel via 2D convolution or difference)
        gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame.astype(float)
        # Approximate discrete Laplacian: L = 4*I(x,y) - I(x+1,y) - I(x-1,y) - I(x,y+1) - I(x,y-1)
        if gray.shape[0] > 4 and gray.shape[1] > 4:
            lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
            blur_var = float(np.var(lap))
        else:
            blur_var = 150.0
        brightness = float(np.mean(gray))
        return blur_var, brightness

    def evaluate_frame(
        self,
        frame_shape: Tuple[int, int],  # (height, width)
        face_bbox: Tuple[float, float, float, float],  # (x_min, y_min, x_max, y_max) in [0, 1]
        head_pose: Tuple[float, float, float],  # (yaw, pitch, roll) in degrees
        left_eye_ear: float,
        right_eye_ear: float,
        left_eye_width_px: float,
        right_eye_width_px: float,
        blur_variance: float,
        mediapipe_confidence: float = 0.95,
        brightness: Optional[float] = None,
        estimated_distance_cm: Optional[float] = None,
        left_eye_occluded: bool = False,
        right_eye_occluded: bool = False,
    ) -> QualityGateResult:
        """Evaluates incoming frame parameters against strict clinical quality bounds."""
        reasons: List[str] = []
        guidance: Optional[str] = None

        h, w = frame_shape
        face_w = max(0.0, (face_bbox[2] - face_bbox[0]) * w)
        face_ratio = face_w / w if w > 0 else 0.0

        # 1. Camera distance / face size
        if face_ratio < self.min_face_ratio:
            reasons.append(f"FACE_TOO_FAR: ratio={face_ratio:.2f} < {self.min_face_ratio}")
            guidance = "Vui lòng ngồi gần camera hơn."
        elif face_ratio > self.max_face_ratio:
            reasons.append(f"FACE_TOO_CLOSE: ratio={face_ratio:.2f} > {self.max_face_ratio}")
            guidance = "Vui lòng ngồi lùi xa camera một chút."

        # Physical distance check if provided
        if estimated_distance_cm is not None:
            if estimated_distance_cm > self.max_distance_cm:
                if not any("FACE_TOO_FAR" in r for r in reasons):
                    reasons.append(f"DISTANCE_TOO_FAR: {estimated_distance_cm:.1f}cm > {self.max_distance_cm}cm")
                    if not guidance:
                        guidance = "Vui lòng ngồi gần camera hơn (khoảng 45-60cm)."
            elif estimated_distance_cm < self.min_distance_cm:
                if not any("FACE_TOO_CLOSE" in r for r in reasons):
                    reasons.append(f"DISTANCE_TOO_CLOSE: {estimated_distance_cm:.1f}cm < {self.min_distance_cm}cm")
                    if not guidance:
                        guidance = "Vui lòng ngồi lùi xa camera một chút (khoảng 45-60cm)."

        # 2. Head pose
        yaw, pitch, roll = head_pose
        if abs(yaw) > self.max_yaw_deg:
            reasons.append(f"HEAD_YAW_EXCEEDED: |yaw|={abs(yaw):.1f}° > {self.max_yaw_deg}°")
            guidance = "Vui lòng nhìn thẳng, không quay mặt sang hai bên."
        if abs(pitch) > self.max_pitch_deg:
            reasons.append(f"HEAD_PITCH_EXCEEDED: |pitch|={abs(pitch):.1f}° > {self.max_pitch_deg}°")
            guidance = "Vui lòng giữ thẳng cằm, không cúi hoặc ngửa đầu."
        if abs(roll) > self.max_roll_deg:
            reasons.append(f"HEAD_ROLL_EXCEEDED: |roll|={abs(roll):.1f}° > {self.max_roll_deg}°")
            guidance = "Vui lòng giữ đầu thẳng, không nghiêng sang vai."

        # 3. Blur
        if blur_variance < self.min_blur_var:
            reasons.append(f"IMAGE_BLURRY: var={blur_variance:.1f} < {self.min_blur_var}")
            guidance = "Camera bị mờ hoặc rung lắc. Vui lòng giữ yên thiết bị."

        # 4. Illumination / Brightness
        if brightness is not None:
            if brightness < self.min_brightness:
                reasons.append(f"LIGHTING_TOO_DARK: val={brightness:.1f} < {self.min_brightness}")
                if not guidance:
                    guidance = "Ánh sáng quá tối. Vui lòng bật thêm đèn hoặc ngồi gần nguồn sáng."
            elif brightness > self.max_brightness:
                reasons.append(f"LIGHTING_TOO_BRIGHT: val={brightness:.1f} > {self.max_brightness}")
                if not guidance:
                    guidance = "Ánh sáng quá chói hoặc lóa. Vui lòng tránh nguồn sáng mạnh chiếu thẳng vào camera."

        # 5. Eye Occlusion
        if left_eye_occluded or right_eye_occluded:
            reasons.append("EYE_OCCLUDED")
            if not guidance:
                guidance = "Mắt bị che khuất hoặc không nhìn rõ."

        # 6. Eye width / resolution
        if left_eye_width_px < self.min_eye_width_px or right_eye_width_px < self.min_eye_width_px:
            reasons.append("EYE_RESOLUTION_TOO_LOW")
            if not guidance:
                guidance = "Độ phân giải mắt quá nhỏ để theo dõi chính xác."

        # 7. Blink detection (EAR below threshold)
        is_blinking = (left_eye_ear < self.min_ear_blink) or (right_eye_ear < self.min_ear_blink)
        if is_blinking:
            reasons.append("EYE_BLINK_DETECTED")

        # Aggregate Camera Quality Score [0, 1]
        cam_components = [
            np.clip(1.0 - abs(face_ratio - 0.40) / 0.30, 0.0, 1.0),
            np.clip(1.0 - abs(yaw) / (self.max_yaw_deg * 2.0), 0.0, 1.0),
            np.clip(1.0 - abs(pitch) / (self.max_pitch_deg * 2.0), 0.0, 1.0),
            np.clip(1.0 - abs(roll) / (self.max_roll_deg * 2.0), 0.0, 1.0),
            np.clip(blur_variance / 200.0, 0.0, 1.0),
        ]
        if brightness is not None:
            bright_score = np.clip(1.0 - abs(brightness - 128.0) / 100.0, 0.0, 1.0)
            cam_components.append(bright_score)

        cam_score = float(np.mean(cam_components))

        # Aggregate Eye Tracking Quality Score [0, 1]
        track_components = [
            float(mediapipe_confidence),
            1.0 if not is_blinking else 0.1,
            np.clip((left_eye_width_px + right_eye_width_px) / 100.0, 0.0, 1.0),
            0.1 if (left_eye_occluded or right_eye_occluded) else 1.0,
        ]
        track_score = float(np.mean(track_components))

        structural_errors = [r for r in reasons if r != "EYE_BLINK_DETECTED"]
        is_acceptable = (len(structural_errors) == 0) and not is_blinking

        return QualityGateResult(
            is_acceptable=is_acceptable,
            camera_quality_score=round(cam_score, 4),
            tracking_quality_score=round(track_score, 4),
            failure_reasons=reasons,
            user_guidance=guidance,
            blur_variance=round(blur_variance, 1),
            brightness=round(brightness, 1) if brightness is not None else None,
            estimated_distance_cm=round(estimated_distance_cm, 1) if estimated_distance_cm is not None else None,
        )

    def evaluate_image_frame(
        self,
        frame: np.ndarray,
        landmark_data: Optional[Any] = None,
    ) -> QualityGateResult:
        """Evaluates an image frame array directly, extracting quality metrics."""
        h, w = frame.shape[:2]
        blur_var, brightness = self.compute_image_metrics(frame)

        if landmark_data is None:
            # If no landmarks provided, evaluate image-level properties only
            reasons = []
            guidance = None
            if blur_var < self.min_blur_var:
                reasons.append(f"IMAGE_BLURRY: var={blur_var:.1f} < {self.min_blur_var}")
                guidance = "Camera bị mờ hoặc rung lắc."
            if brightness < self.min_brightness:
                reasons.append(f"LIGHTING_TOO_DARK: val={brightness:.1f}")
                guidance = "Ánh sáng quá tối."
            elif brightness > self.max_brightness:
                reasons.append(f"LIGHTING_TOO_BRIGHT: val={brightness:.1f}")
                guidance = "Ánh sáng quá chói hoặc lóa."

            reasons.append("NO_FACE_DETECTED")
            return QualityGateResult(
                is_acceptable=False,
                camera_quality_score=0.3,
                tracking_quality_score=0.0,
                failure_reasons=reasons,
                user_guidance=guidance or "Không phát hiện khuôn mặt trong khung hình.",
                blur_variance=round(blur_var, 1),
                brightness=round(brightness, 1),
            )

        return self.evaluate_frame(
            frame_shape=(h, w),
            face_bbox=landmark_data.face_bbox,
            head_pose=landmark_data.head_pose,
            left_eye_ear=landmark_data.left_ear,
            right_eye_ear=landmark_data.right_ear,
            left_eye_width_px=landmark_data.left_eye_width_px,
            right_eye_width_px=landmark_data.right_eye_width_px,
            blur_variance=blur_var,
            mediapipe_confidence=landmark_data.confidence,
            brightness=brightness,
            estimated_distance_cm=landmark_data.estimated_distance_cm,
            left_eye_occluded=not landmark_data.left_eye_valid,
            right_eye_occluded=not landmark_data.right_eye_valid,
        )


