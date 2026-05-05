"""
Face and eye detection utilities for real-time drowsiness detection.

Two complementary methods are provided:

1. **OpenCV Haar Cascades** — lightweight, no extra dependencies, works
   offline. Used as the primary detector.
2. **Eye Aspect Ratio (EAR)** — geometry-based blink/drowsiness metric
   derived from facial landmarks (requires ``dlib`` or ``mediapipe``).

The EAR approach alone can detect blinks (EAR drops sharply) but misses
subtle drooping. Combining EAR with the CNN classifier gives a more robust
system.
"""

import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# EAR computation
# ---------------------------------------------------------------------------

def compute_ear(eye_landmarks: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a set of 6 eye landmarks.

    The EAR is defined as::

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    where p1..p6 are the six eye landmark coordinates ordered as:
    p1 (left corner), p2 (upper-left), p3 (upper-right),
    p4 (right corner), p5 (lower-right), p6 (lower-left).

    Args:
        eye_landmarks (np.ndarray): Array of shape ``(6, 2)`` with ``(x, y)``
                                    coordinates of the six eye landmarks.

    Returns:
        float: EAR value. Values below 0.25 typically indicate a closed eye.
               Returns ``0.0`` if computation fails.

    References:
        Soukupová & Čech (2016), "Real-Time Eye Blink Detection using Facial
        Landmarks", CVWW.
    """
    if eye_landmarks is None or len(eye_landmarks) != 6:
        return 0.0

    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


# ---------------------------------------------------------------------------
# Haar Cascade detector
# ---------------------------------------------------------------------------

class FaceEyeDetector:
    """
    Detect faces and eyes from video frames using OpenCV Haar Cascades.

    This detector is dependency-free (requires only ``opencv-python``) and
    runs in real-time on CPU.

    Args:
        face_cascade_path (str | None): Path to the face Haar cascade XML.
            If ``None``, uses OpenCV's bundled ``haarcascade_frontalface_default.xml``.
        eye_cascade_path (str | None): Path to the eye Haar cascade XML.
            If ``None``, uses OpenCV's bundled ``haarcascade_eye.xml``.
        scale_factor (float): Image pyramid scale factor for detection.
        min_neighbors (int): Minimum neighbours for a detection to be kept.
        min_face_size (tuple): Minimum ``(w, h)`` in pixels for face detection.

    Example:
        >>> detector = FaceEyeDetector()
        >>> faces, eyes_list = detector.detect(frame)
        >>> for (x, y, w, h) in faces:
        ...     face_roi = frame[y:y+h, x:x+w]
    """

    # Default cascade paths shipped with OpenCV
    _DEFAULT_FACE = "haarcascade_frontalface_default.xml"
    _DEFAULT_EYE = "haarcascade_eye.xml"

    def __init__(
        self,
        face_cascade_path=None,
        eye_cascade_path=None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: tuple = (80, 80),
    ):
        cv_data = Path(cv2.data.haarcascades)

        face_xml = face_cascade_path or str(cv_data / self._DEFAULT_FACE)
        eye_xml = eye_cascade_path or str(cv_data / self._DEFAULT_EYE)

        self.face_cascade = cv2.CascadeClassifier(face_xml)
        self.eye_cascade = cv2.CascadeClassifier(eye_xml)

        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load face cascade from '{face_xml}'. "
                "Please install opencv-python or provide a valid path."
            )

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray):
        """
        Detect faces and eyes in a BGR frame.

        Args:
            frame (np.ndarray): BGR image array (from ``cv2.VideoCapture``).

        Returns:
            tuple:
                - **faces** (list): List of ``(x, y, w, h)`` tuples for each
                  detected face.
                - **eyes_per_face** (list[list]): Each element is a list of
                  ``(x, y, w, h)`` tuples for the eyes found within the
                  corresponding face ROI. Coordinates are relative to the
                  full frame (not the face ROI).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
        )
        faces = list(faces) if len(faces) > 0 else []

        eyes_per_face = []
        for fx, fy, fw, fh in faces:
            face_gray = gray[fy: fy + fh, fx: fx + fw]
            eyes_raw = self.eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20),
            )
            # Convert eye coordinates to full-frame coordinates
            eyes_abs = []
            if len(eyes_raw) > 0:
                for ex, ey, ew, eh in eyes_raw:
                    eyes_abs.append((fx + ex, fy + ey, ew, eh))
            eyes_per_face.append(eyes_abs)

        return faces, eyes_per_face

    def extract_eye_roi(
        self,
        frame: np.ndarray,
        eye_rect: tuple,
        target_size: tuple = (64, 64),
    ) -> np.ndarray:
        """
        Crop and resize a single eye region for model inference.

        Args:
            frame (np.ndarray): Full BGR frame.
            eye_rect (tuple): ``(x, y, w, h)`` of the eye region in the frame.
            target_size (tuple): ``(width, height)`` to resize the crop to.

        Returns:
            np.ndarray: Normalized float32 array of shape
                        ``(1, H, W, 3)`` ready for ``model.predict()``.
        """
        x, y, w, h = eye_rect
        # Add a small margin around the eye
        margin = int(min(w, h) * 0.15)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        eye_crop = frame[y1:y2, x1:x2]
        if eye_crop.size == 0:
            return None

        eye_resized = cv2.resize(eye_crop, target_size)
        eye_rgb = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2RGB)
        eye_norm = eye_rgb.astype(np.float32) / 255.0
        return np.expand_dims(eye_norm, axis=0)  # (1, H, W, 3)

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        faces: list,
        eyes_per_face: list,
        drowsy: bool = False,
        score: float = 0.0,
    ) -> np.ndarray:
        """
        Draw bounding boxes and status overlay on a frame.

        Args:
            frame (np.ndarray): BGR frame to annotate (modified in-place).
            faces (list): Face rectangles from ``detect()``.
            eyes_per_face (list): Eye rectangles per face from ``detect()``.
            drowsy (bool): Whether the driver is classified as drowsy.
            score (float): Drowsiness probability (0–1).

        Returns:
            np.ndarray: Annotated BGR frame.
        """
        face_color = (0, 0, 255) if drowsy else (0, 255, 0)
        status_text = "DROWSY ⚠️" if drowsy else "Awake ✅"
        score_text = f"Score: {score:.2f}"

        for (x, y, w, h), eyes in zip(faces, eyes_per_face):
            cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

        # Status overlay
        overlay_color = (0, 0, 200) if drowsy else (0, 150, 0)
        cv2.rectangle(frame, (0, 0), (280, 60), overlay_color, -1)
        cv2.putText(
            frame, status_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, score_text, (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )
        return frame
