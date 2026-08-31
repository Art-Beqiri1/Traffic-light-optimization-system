# FlowMind Live - four cameras and operator controls

This update adds yellow lights, up to four camera feeds for **one intersection**,
and AI / manual / fixed-timer modes in both the web and desktop apps.

## Start

From the extracted `flowmindrl` directory:

```sh
pip install -r requirements.txt
python desktop_app.py
```

Or use the browser dashboard:

```sh
python live_app.py
```

On Windows, after installing dependencies, you can also use `RUN_LIVE_DESKTOP.bat`
or `RUN_LIVE_WEB.bat`. The original `install_and_run.bat` still launches the
legacy standalone detector (`app.py`), which does not contain these live features.

Open http://127.0.0.1:5000. Run **one app at a time**: separate processes do not
share cameras, controller state or config writes. The desktop app shows decoded
frames directly; the browser uses per-camera WebRTC with MJPEG fallback.

If optional `aiortc` / `av` cannot install or import, remove their requirement
lines and use MJPEG or the desktop app. Neither fixed/manual control nor the
Max-Pressure bootstrap needs PyTorch; a missing DQN dependency is reported in
the training status instead of preventing those controls from starting.

## 1. Yellow-light sequence

Every live signal change follows:

**N/S green → N/S yellow → all red → E/W green** (and the reverse).

- Defaults: **3 seconds yellow**, **2 seconds all red**.
- Start-up begins with all red; no immediate conflicting green.
- The opposite direction remains red during yellow.
- The live controller uses elapsed wall-clock time, independently of video FPS.
- A delayed control tick cannot skip a clearance stage.
- Editing timers does not shorten an already-active yellow or all-red interval.
- Live signal colors appear in the browser, desktop and processed video overlay.
- Video can freeze if capture/detection stalls; use the dedicated signal panel
  and camera health status to distinguish the current command from old footage.

These are **demo timing defaults**, not site-specific road engineering values.

## 2. Connect up to four cameras

1. Enter an RTSP URL, local video path, or webcam index such as `0` in each slot.
2. Assign incoming **North, South, East or West** to each active camera.
3. Leave unused URLs blank, or uncheck Enabled.
4. Click **Save & connect**. Each camera also has its own Start / Stop buttons.

All four feeds have independent capture threads and YOLO/ByteTrack instances.
Counts map to the assigned approach at intersection **0-0**, without summing the
same approach from multiple cameras. The other displayed intersections remain
simulated. Quality settings apply to all camera slots.

For approach cameras the full frame is the counting zone. Frame each camera to
cover only its intended incoming lane(s); this remains a simplified vehicle
occupancy estimate, not a calibrated stopped-vehicle queue measurement. Do not
point overlapping cameras at the same traffic and expect correct totals.

**Legacy single-camera setup:** old `rtsp_url` configurations migrate to camera
N with "Whole intersection" selected, preserving the original quadrant zones.
This option must be used alone. For four views, change it to North and assign
the remaining three approaches. All active approach assignments must be unique.

Changing a source replaces its tracker and reconnects it; previous source
threads cannot publish into the replacement camera's state. Changing shared
quality settings restarts active camera pipelines to apply them consistently.
Tracking totals reset on replacement. Four models require more CPU/GPU and RAM
than one; start with the included nano model at 640 and measure your hardware.

## 3. Manual and fixed-timer operation

Use the mode buttons in the signal/control panel:

| Mode | Behavior |
| --- | --- |
| AI / retry AI | Max-Pressure initially, then Double DQN when training succeeds. |
| Switch to manual | Holds the current green at camera node 0-0, or requests all red if it is already transitioning. |
| Fixed timer | Alternates N/S and E/W using the configured green durations, without AI decisions. |

In manual mode choose **N/S green**, **E/W green**, or **Hold all red**. A requested
phase is shown separately from the currently displayed lights. Manual green
holds until another command; it does not automatically expire at 60 seconds.
Manual opposite-green requests respect a 7-second minimum green. Hold all red
starts yellow immediately, then holds red after clearance. No command can skip
an active yellow or all-red stage. Opposing greens cannot be selected together.

Mode and timer controls apply to **all displayed intersections**. The four
camera feeds still belong to one intersection, not four separate junctions.

Fixed-timer defaults are **25s N/S green and 25s E/W green**. Green durations can
be set independently from 7–60s; yellow and all-red from 1–10s. These validation
ranges are demo constraints, not a statement that every setting is road-safe.
The cycle includes both yellow and all-red intervals. Switching into fixed mode
uses the current green's elapsed time, so it may begin clearance promptly if the
new duration has already elapsed.

Camera settings, selected mode, and timers persist in `live_config.json`.
An active manual green is **not restored after restart**: manual mode restarts
holding all red. Runtime fallback notices are not persisted across restart.

## Automatic fallback

When AI mode is selected, these faults latch **fixed-timer fallback**:

- AI decision raises an exception or returns an invalid/incomplete action.
- AI inference takes 3 seconds or longer.
- An enabled, configured camera or its detector becomes unavailable/stale.
  Successful detection must be from a frame captured within the last 5 seconds.
  There is an 8-second connection/model startup grace period; if loading takes
  longer the fallback remains active until the operator retries AI.

A healthy empty frame (zero vehicles) is valid. A detection failure is not
silently interpreted as zero demand. Stale camera counts are excluded. Stopping
an enabled, configured camera is a health fault; disable it to intentionally
exclude that slot.

Valid but poor AI decisions cannot all be detected automatically; operator
controls remain available.

Fallback is visible and **does not silently switch back to AI**. Resolve the
problem, then select **AI / retry AI**. A background training completion cannot
override manual mode or a fallback. Failed training alone leaves the existing
Max-Pressure bootstrap available; it is reported in the training status.

AI work runs on a copy of the simulated network in a separate thread, with at
most one outstanding worker. A hung worker cannot accumulate replacement
threads. Control commands and the fixed-timer loop do not wait for inference.
This cannot guarantee recovery from a process crash, machine freeze, power loss,
or a native extension that blocks the entire process. A timed-out inference
thread cannot be forcibly killed; if it never returns, keep fixed/manual mode
or restart the app before retrying AI.

## API

- `GET /api/state`: `cameras` (four slots, health/counts), aggregate `vision`,
  `traffic.camera_signal`, `traffic.control`, settings and simulated efficiency.
- `POST /api/control`: `mode` (`ai`, `manual`, `fixed`), optional
  `manual_phase` (`NS_GREEN`, `EW_GREEN`, `ALL_RED`), `fixed_ns_seconds`,
  `fixed_ew_seconds`, `yellow_seconds`, `all_red_seconds`. Timer values are integers.
  Manual commands require manual mode; validation failures return HTTP 400.
- `POST /api/settings`: `cameras` array (at most four), each with `id`
  (`N`, `S`, `E`, `W`), `approach`, `rtsp_url`, `enabled`; original quality fields
  and the legacy single `rtsp_url` request are also supported.
- `POST /api/camera/start` or `/stop`: all configured cameras, or append
  `?camera_id=N` (likewise S/E/W) for one slot.
- `GET /video_feed?camera_id=N`: per-camera MJPEG.
- `POST /offer`: SDP offer with `camera_id` for per-camera WebRTC.

The browser marks signal state **UNKNOWN** when API updates stop. Browser
connection loss does not stop a running backend timer, but a backend/process
failure does. MJPEG encoding and camera connect/stop work run outside the HTTP
event loop so they do not block light commands.

## Scope and security

This is an **intersection visualization / research prototype**. It has no
physical traffic-light driver, certified conflict monitor, turn-phase handling,
pedestrian sequencing, cabinet integration, or independent hardware watchdog.
Do not connect it directly to public-road signals. Real deployment requires a
qualified traffic engineer, a certified controller and independent fail-safe
hardware. Manual/fixed fallback here protects demo operation, not public roads.

Efficiency values come from the simulator, including synthetic neighboring
intersections; they are not measured real-world waiting-time guarantees.
Offline training/benchmark timing remains unchanged. Live yellow timing changes
are applied through an explicit external-signal path in `network_env.py`.

The web server binds to **127.0.0.1 by default**. It has no authentication and
stores camera credentials in local JSON. Do not publish it or expose its port
to an untrusted network. For intentional trusted-LAN use set `FLOWMIND_HOST`
(e.g. `0.0.0.0`), with suitable network restrictions and authenticated access.
Cross-origin browser commands are rejected; this does not replace authentication.

## Validation

```sh
python -m unittest test_live_features -v
```

The suite covers clearance timings, rapid manual requests, unequal fixed
intervals, mode transitions, invalid commands, camera mapping, stale detection,
source replacement, AI errors/hangs and HTTP commands with four MJPEG clients.
Camera/inference tests use substitutes where physical/model inputs are needed.
Original benchmark dynamics were separately compared against the supplied
archive over 1,000 identical seeded steps with matching results.

Real RTSP cameras, camera placement, full YOLO accuracy/performance and DQN
training are not verified by these tests. Test your local setup before relying
on the demo. The bundled source is not a compiled Windows executable.
