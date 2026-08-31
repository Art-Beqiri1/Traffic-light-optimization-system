"""
webrtc_stream.py — Low-latency WebRTC delivery of the processed camera
feed, replacing the old MJPEG-over-HTTP approach.

Why WebRTC and not HLS: HLS is chunk-based (typically 2-6s of latency
even in "low-latency" mode) because it's designed for scalable one-to-
many broadcast, not real-time interaction. WebRTC is designed for
sub-second glass-to-glass latency, which is what you want when a human
is watching this feed to verify the sensor is seeing what it should.

Important: this only affects the HUMAN-FACING monitoring view. The
actual traffic-light decision logic in live_controller.py reads
directly from vision.py's shared state in the backend — it never
waits on this video track or on anyone's browser. Switching to
WebRTC does not change how fast the light-switching logic reacts;
it changes how fast a person watching the dashboard sees what the
sensor sees.

LiveDetectionTrack always sends the MOST RECENT processed frame at
call time (not a queued/buffered one), so if the network can't keep
up with the true detection rate you get frame drops (repeats), never
growing latency from a backlog.
"""
from __future__ import annotations

import time

import av
import cv2
import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE


class LiveDetectionTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, vision, fps: int = 20):
        super().__init__()
        self.vision = vision
        self._frame_interval = 1.0 / fps
        self._start = None
        self._timestamp = 0
        self._placeholder = None

    def _get_placeholder(self):
        if self._placeholder is None:
            frame = np.zeros((240, 320, 3), dtype="uint8")
            cv2.putText(frame, "waiting for camera...", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self._placeholder = frame
        return self._placeholder

    async def recv(self):
        # pace to target fps using the same clock convention as aiortc's
        # own VideoStreamTrack, but WITHOUT waiting on a new frame — we
        # always grab whatever the latest processed frame is right now
        if self._start is None:
            self._start = time.time()
        self._timestamp += int(VIDEO_CLOCK_RATE * self._frame_interval)
        wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
        if wait > 0:
            import asyncio
            await asyncio.sleep(wait)

        vision = self.vision() if callable(self.vision) else self.vision
        frame = vision.get_latest_frame() if vision._running else None
        if frame is None:
            frame = self._get_placeholder()

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self._timestamp
        video_frame.time_base = VIDEO_TIME_BASE
        return video_frame
