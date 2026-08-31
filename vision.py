"""One camera pipeline: capture, YOLO/ByteTrack, queue zones and signal overlay.
CameraManager supplies either one full-frame approach zone or the legacy
whole-intersection quadrant zones. Detection timestamps expose stale frames.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - only hit if ultralytics isn't installed
    YOLO = None

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

DEFAULT_ZONES = {
    "N": (0.0, 0.0, 1.0, 0.35),   # x1, y1, x2, y2 as fractions of frame size
    "S": (0.0, 0.65, 1.0, 1.0),
    "E": (0.65, 0.0, 1.0, 1.0),
    "W": (0.0, 0.0, 0.35, 1.0),
}


@dataclass
class VisionState:
    """Snapshot the dashboard/API reads. Updated in place by the capture
    thread under `lock`."""
    camera_status: str = "stopped"
    last_error: str = ""
    rtsp_url: str = ""
    vehicle_counts: Dict[str, int] = field(
        default_factory=lambda: {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0})
    total_count: int = 0
    in_count: int = 0
    out_count: int = 0
    queues: Dict[str, int] = field(default_factory=lambda: {"N": 0, "S": 0, "E": 0, "W": 0})
    detection_ok: bool = False
    last_detection_at: float = 0.0
    fps: float = 0.0
    capture_fps: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "camera_status": self.camera_status,
                "last_error": self.last_error,
                "rtsp_url": self.rtsp_url,
                "vehicle_counts": dict(self.vehicle_counts),
                "total_count": self.total_count,
                "in_count": self.in_count,
                "out_count": self.out_count,
                "queues": dict(self.queues),
                "detection_ok": self.detection_ok,
                "last_detection_age_s": round(max(0, time.monotonic() - self.last_detection_at), 2) if self.last_detection_at else None,
                "fps": round(self.fps, 1),
                "capture_fps": round(self.capture_fps, 1),
            }


class VehicleVision:
    """Owns the camera, the YOLO model, and the background capture thread.

    Usage:
        vision = VehicleVision(rtsp_url="rtsp://user:pass@host/stream1")
        vision.start()
        ... vision.state.snapshot() for JSON, vision.mjpeg_frames() for /video_feed ...
        vision.stop()
    """

    def __init__(self, rtsp_url: str = "", confidence: float = 0.35,
                 line_position: float = 0.55, zones: Optional[dict] = None,
                 model_name: str = "yolov8n.pt", imgsz: int = 640,
                 rtsp_transport: str = "tcp", enhance: bool = True,
                 tracker_config: Optional[str] = None,
                 get_signal_state=None):
        self.state = VisionState(rtsp_url=rtsp_url)
        self.confidence = confidence
        self.line_position = line_position
        self.zones = zones or DEFAULT_ZONES
        self.model_name = model_name
        self.imgsz = imgsz
        self.rtsp_transport = rtsp_transport
        self.enhance = enhance
        # Tuned ByteTrack config (see tracker_tuned.yaml) — keeps tracks
        # alive through brief occlusion/low-confidence frames instead of
        # dropping vehicles mid-track, at a small cost of more ID
        # switches. Falls back to ultralytics' bundled "bytetrack.yaml"
        # if a custom path isn't found.
        default_tracker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "tracker_tuned.yaml")
        self.tracker_config = tracker_config or (
            default_tracker if os.path.exists(default_tracker) else "bytetrack.yaml")
        # optional callable -> {"ns": "GREEN"/"RED", "ew": "GREEN"/"RED"}
        # wired up by live_app.py so the recommended signal (computed by
        # live_controller.py from THIS camera's queues) can be burned
        # directly into the video overlay, not just shown elsewhere on
        # the page.
        self.get_signal_state = get_signal_state

        self._camera = None
        self._model = None
        self._model_lock = threading.Lock()
        self._model_name_loaded = None
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame = None
        self._frame_version = 0
        self._new_frame_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._previous_y: Dict[int, int] = {}
        self._counted_ids = set()
        self._is_file_source = False
        self._file_frame_interval = None
        # raw (undetected) frame handoff between capture and process threads
        self._raw_frame = None
        self._raw_frame_at = 0.0
        self._raw_frame_lock = threading.Lock()
        self._raw_frame_event = threading.Event()
        self._capture_frame_count = 0
        self._capture_fps_t = time.time()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def set_rtsp_url(self, url: str):
        with self.state.lock:
            self.state.rtsp_url = url

    def start(self):
        if self._running:
            return
        if any(t is not None and t.is_alive() for t in (self._capture_thread, self._process_thread)):
            raise RuntimeError("Camera is still stopping; try again shortly")
        self._raw_frame = None
        self._raw_frame_event.clear()
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def stop(self):
        self._running = False
        self._raw_frame_event.set()  # wake the process thread so it can exit
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3)
        if self._process_thread is not None:
            self._process_thread.join(timeout=3)
        with self.state.lock:
            self.state.camera_status = "stopped"
            self.state.detection_ok = False
            self.state.queues = dict.fromkeys(("N", "S", "E", "W"), 0)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _load_model(self):
        if YOLO is None:
            with self.state.lock:
                self.state.last_error = "ultralytics is not installed"
            return False
        if self._model is not None and self._model_name_loaded == self.model_name:
            return True
        try:
            with self._model_lock:
                self._model = YOLO(self.model_name)
                self._model_name_loaded = self.model_name
            return True
        except Exception as error:
            with self.state.lock:
                self.state.last_error = f"YOLO load failed: {error}"
            return False

    def _open_camera(self, url: str) -> bool:
        self._release_camera()
        if not url:
            return False
        source = int(url) if url.isdigit() else url
        if isinstance(source, str) and source.startswith("rtsp://"):
            # Force TCP transport by default (fixes most UDP-packet-loss
            # pixelation), AND tell ffmpeg's demuxer not to buffer ahead
            # (nobuffer/low_delay/max_delay=0). Without these, ffmpeg can
            # accumulate a growing backlog of undelivered frames on its
            # own, independent of anything happening on the Python side.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.rtsp_transport}"
                "|stimeout;5000000|buffer_size;1024000"
                "|fflags;nobuffer|flags;low_delay|max_delay;0"
            )
        cam = cv2.VideoCapture(source) if isinstance(source, int) else cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cam.isOpened():
            cam.release()
            return False
        self._camera = cam
        # A local video file (real recorded footage, not a live RTSP/webcam
        # source) should loop seamlessly for testing instead of going
        # through the "disconnected, reconnecting" retry path meant for
        # flaky live cameras.
        self._is_file_source = isinstance(source, str) and os.path.exists(source)
        if self._is_file_source:
            # A file has no natural real-time pacing the way a live camera
            # does — cv2 will happily hand back frames as fast as it can
            # decode them (thousands of fps for a small test clip), which
            # just burns CPU pointlessly and isn't representative of how
            # this behaves against a real camera. Pace it to the file's
            # own frame rate so local testing behaves like the real thing.
            file_fps = cam.get(cv2.CAP_PROP_FPS)
            self._file_frame_interval = (1.0 / file_fps) if file_fps and file_fps > 0 else 1.0 / 25
        return True

    def _release_camera(self):
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
        self._camera = None

    def _zone_box(self, w: int, h: int, key: str):
        x1f, y1f, x2f, y2f = self.zones.get(key, DEFAULT_ZONES[key])
        return int(x1f * w), int(y1f * h), int(x2f * w), int(y2f * h)

    def _in_zone(self, cx: int, cy: int, box) -> bool:
        x1, y1, x2, y2 = box
        return x2 > x1 and y2 > y1 and x1 <= cx <= x2 and y1 <= cy <= y2

    # ------------------------------------------------------------------
    # capture thread: pulls frames from the camera as fast as the source
    # provides them, and ONLY ever keeps the newest one. This is the fix
    # for a growing delay over time — if the old code's single loop did
    # read -> detect -> read -> detect, and detection was slower than the
    # camera's actual frame rate, the video source fell further and
    # further behind every single iteration (a 15s lag is exactly what
    # that pattern produces after enough minutes of runtime). Decoupling
    # capture from processing means detection always works on the FRESHEST
    # frame available, and simply drops whatever it didn't have time to
    # look at — latency stays bounded instead of growing without limit.
    # ------------------------------------------------------------------
    def _capture_loop(self):
        while self._running:
            with self.state.lock:
                url = self.state.rtsp_url

            if not url:
                with self.state.lock:
                    self.state.camera_status = "no camera URL configured"
                time.sleep(1)
                continue

            if self._camera is None:
                with self.state.lock:
                    self.state.camera_status = "connecting..."
                if not self._open_camera(url):
                    with self.state.lock:
                        self.state.camera_status = "connection failed, retrying"
                    time.sleep(3)
                    continue
                with self.state.lock:
                    self.state.camera_status = "online"

            success, frame = (False, None)
            try:
                success, frame = self._camera.read()
            except Exception as error:
                with self.state.lock:
                    self.state.last_error = str(error)

            if not success or frame is None:
                if self._is_file_source and self._camera is not None:
                    self._camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    with self.state.lock:
                        self.state.camera_status = "online (looping test footage)"
                    continue
                with self.state.lock:
                    self.state.camera_status = "disconnected, reconnecting"
                self._release_camera()
                time.sleep(2)
                continue

            with self._raw_frame_lock:
                self._raw_frame = frame
                self._raw_frame_at = time.monotonic()
            self._raw_frame_event.set()

            self._capture_frame_count += 1
            now = time.time()
            if now - self._capture_fps_t >= 1.0:
                with self.state.lock:
                    self.state.capture_fps = self._capture_frame_count / (now - self._capture_fps_t)
                self._capture_frame_count = 0
                self._capture_fps_t = now

            if self._is_file_source and self._file_frame_interval:
                time.sleep(self._file_frame_interval)

        self._release_camera()

    def _process_loop(self):
        if not self._load_model():
            with self.state.lock:
                self.state.camera_status = "model load failed"
        last_fps_t = time.time()
        frame_count = 0

        while self._running:
            if self._model_name_loaded != self.model_name:
                self._load_model()

            # block briefly for a new raw frame instead of busy-looping;
            # if none shows up (camera not connected yet), just re-check
            got_new = self._raw_frame_event.wait(timeout=0.5)
            self._raw_frame_event.clear()
            if not got_new:
                continue

            with self._raw_frame_lock:
                frame = None if self._raw_frame is None else self._raw_frame.copy()
                captured_at = self._raw_frame_at
            if frame is None:
                continue

            try:
                frame = self._process_frame(frame, captured_at)
            except Exception as error:
                with self.state.lock:
                    self.state.detection_ok = False
                    self.state.last_error = f"processing error: {type(error).__name__}"
                continue

            with self._frame_lock:
                self._latest_frame = frame
                self._frame_version += 1
            self._new_frame_event.set()

            frame_count += 1
            now = time.time()
            if now - last_fps_t >= 1.0:
                with self.state.lock:
                    self.state.fps = frame_count / (now - last_fps_t)
                frame_count = 0
                last_fps_t = now

    def _enhance(self, frame):
        """Cheap, fast clean-up for a noisy/pixelated feed: CLAHE contrast
        boost on luminance (helps low-light macroblocking read as
        vehicles instead of noise) + a mild unsharp mask (helps thin
        vehicle edges survive compression blur). Both run in well under
        a millisecond on a 720p frame — this is not the same as denoising
        video after the fact, it's meant to counteract exactly the kind
        of softness/blockiness a compressed RTSP stream introduces."""
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.2)
            frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        except Exception:
            pass
        return frame

    def _process_frame(self, frame, captured_at=None):
        detection_ok = False
        if self.enhance:
            frame = self._enhance(frame)

        h, w = frame.shape[:2]
        zone_boxes = {k: self._zone_box(w, h, k) for k in ("N", "S", "E", "W")}
        live_queue = {"N": 0, "S": 0, "E": 0, "W": 0}

        line_y = int(h * self.line_position)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

        for key, (x1, y1, x2, y2) in zone_boxes.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 1)
            cv2.putText(frame, key, (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 180, 0), 1)

        if self._model is not None:
            try:
                with self._model_lock:
                    results = self._model.track(
                        frame, persist=True, tracker=self.tracker_config,
                        classes=list(VEHICLE_CLASSES.keys()),
                        conf=self.confidence, imgsz=self.imgsz,
                        max_det=200, verbose=False)
                detection_ok = True
                result = results[0] if results else None
                boxes = result.boxes if result is not None else None

                if boxes is not None:
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        vtype = VEHICLE_CLASSES.get(cls_id)
                        if vtype is None:
                            continue
                        track_id = None
                        if boxes.id is not None:
                            try:
                                track_id = int(boxes.id[i].cpu().numpy())
                            except Exception:
                                track_id = None

                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{vtype} {track_id if track_id is not None else ''}"
                        cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        for key, box in zone_boxes.items():
                            if self._in_zone(cx, cy, box):
                                live_queue[key] += 1

                        if track_id is not None:
                            prev_y = self._previous_y.get(track_id)
                            if prev_y is not None:
                                crossed_down = prev_y < line_y <= cy
                                crossed_up = prev_y > line_y >= cy
                                if (crossed_down or crossed_up) and track_id not in self._counted_ids:
                                    self._counted_ids.add(track_id)
                                    with self.state.lock:
                                        self.state.vehicle_counts[vtype] = \
                                            self.state.vehicle_counts.get(vtype, 0) + 1
                                        self.state.total_count += 1
                                        if crossed_down:
                                            self.state.in_count += 1
                                        else:
                                            self.state.out_count += 1
                            self._previous_y[track_id] = cy
            except Exception as error:
                detection_ok = False
                with self.state.lock:
                    self.state.last_error = f"detection error: {type(error).__name__}"

        with self.state.lock:
            self.state.detection_ok = detection_ok
            if detection_ok:
                self.state.last_detection_at = captured_at if captured_at is not None else time.monotonic()
                self.state.last_error = ""
            self.state.queues = live_queue
            status = self.state.camera_status

        overlay_h = 30
        cv2.rectangle(frame, (0, 0), (w, overlay_h), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status}  N:{live_queue['N']} S:{live_queue['S']} "
                            f"E:{live_queue['E']} W:{live_queue['W']}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Burn the current signal recommendation straight into the feed,
        # so "should the light be red or green right now" is answered on
        # the video itself, not just elsewhere on the page.
        if self.get_signal_state is not None:
            try:
                signal = self.get_signal_state()
            except Exception:
                signal = None
            if signal:
                bar_y = overlay_h
                cv2.rectangle(frame, (0, bar_y), (w, bar_y + 26), (0, 0, 0), -1)
                colors = {"GREEN": (0, 210, 0), "YELLOW": (0, 210, 255), "RED": (0, 0, 220)}
                ns_color = colors.get(signal.get("ns"), colors["RED"])
                ew_color = colors.get(signal.get("ew"), colors["RED"])
                cv2.circle(frame, (16, bar_y + 13), 8, ns_color, -1)
                cv2.putText(frame, f"NS {signal.get('ns','?')}", (30, bar_y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ns_color, 1)
                cv2.circle(frame, (150, bar_y + 13), 8, ew_color, -1)
                cv2.putText(frame, f"EW {signal.get('ew','?')}", (164, bar_y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ew_color, 1)
        return frame

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def mjpeg_frames(self):
        """Generator yielding multipart/x-mixed-replace JPEG chunks.

        Streams a NEW frame as soon as one is ready (event-driven, not on
        a fixed timer) so the stream runs at whatever rate detection is
        actually producing frames at, instead of being artificially
        capped below that. JPEG quality is set high (92) since the
        detection overlay text/boxes get visibly blocky at default
        compression, especially on smaller/distant vehicles.
        """
        placeholder = None
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        last_version = -1
        while True:
            got_new = self._new_frame_event.wait(timeout=0.5)
            self._new_frame_event.clear()

            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
                version = self._frame_version

            if frame is None:
                if placeholder is None:
                    import numpy as np
                    placeholder = np.zeros((240, 320, 3), dtype="uint8")
                    cv2.putText(placeholder, "waiting for camera...", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                frame = placeholder
            elif version == last_version and got_new:
                # shouldn't normally happen, but avoid re-sending stale data
                continue
            last_version = version

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    def live_queues(self) -> Dict[str, int]:
        with self.state.lock:
            return dict(self.state.queues)

    def get_latest_frame(self):
        """Returns the newest processed (detection-overlaid) frame as a
        raw BGR numpy array, or None if nothing's been processed yet.

        This is what desktop_app.py uses to display video directly —
        no JPEG encode, no HTTP/WebRTC hop. That's the whole point: for
        a local viewer, this is the only genuinely lossless, lowest-
        latency path, since every other delivery method in this project
        (MJPEG, WebRTC) necessarily re-encodes the frame for transport.
        """
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()
