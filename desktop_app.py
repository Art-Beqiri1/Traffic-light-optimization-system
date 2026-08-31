"""FlowMind Live native desktop app: four direct camera views and shared controls.
Run python desktop_app.py OR live_app.py; separate processes have separate
controllers and config writers. Demo only; no hardware I/O.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import queue

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QPushButton, QComboBox, QCheckBox, QGroupBox, QScrollArea,
    QSizePolicy, QSpinBox, QLayout,
)

from camera_manager import CameraManager
from live_controller import LiveTrafficController
from live_settings import load_config, save_config, update_settings

MODEL_CHOICES = {"Fast (nano)": "yolov8n.pt", "Balanced (small)": "yolov8s.pt", "Accurate (medium)": "yolov8m.pt"}

DARK_STYLE = """
QWidget { background: #0b0f14; color: #e7edf5; font-family: -apple-system, Segoe UI, sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #24303f; border-radius: 8px; margin-top: 14px; padding-top: 10px; font-weight: 600; color: #8ea0b5; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLineEdit, QComboBox, QSpinBox { background: #17202c; border: 1px solid #24303f; border-radius: 6px; padding: 6px; color: #e7edf5; }
QPushButton { background: #4fd1ff; color: #05131a; border: none; border-radius: 6px; padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background: #6fdcff; }
QPushButton:disabled { background:#17202c; color:#637182; border:1px solid #24303f; }
QPushButton:checked { border:2px solid #e7edf5; }
QPushButton#secondary { background: #17202c; color: #e7edf5; border: 1px solid #24303f; }
QLabel#statVal { font-size: 20px; font-weight: 700; }
QLabel#statLbl { font-size: 11px; color: #8ea0b5; }
"""


class SignalLight(QLabel):
    """A single big red/green dot, matching the web dashboard's Signal panel."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(46, 46)
        self.set_state("RED")

    def set_state(self, state: str):
        color = {"GREEN":"#35d07f", "YELLOW":"#ffd43b", "RED":"#ff5c5c"}.get(state, "#8ea0b5")
        self.setStyleSheet(
            f"background: {color}; border-radius: 23px; border: 3px solid {color};")


class IntersectionCard(QWidget):
    """Mirrors the web dashboard's per-intersection grid cell."""
    def __init__(self):
        super().__init__()
        self.setObjectName("intersectionCard")
        self.setStyleSheet("QWidget#intersectionCard { background: #17202c; border: 1px solid #24303f; border-radius: 10px; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.id_label = QLabel("Node")
        self.id_label.setStyleSheet("color: #8ea0b5; font-size: 11px;")
        lights_row = QHBoxLayout()
        self.dot_ns = QLabel(); self.dot_ns.setFixedSize(14, 14)
        self.dot_ew = QLabel(); self.dot_ew.setFixedSize(14, 14)
        lights_row.addWidget(self.dot_ns)
        lights_row.addWidget(self.dot_ew)
        lights_row.addStretch()
        self.queue_label = QLabel("")
        self.queue_label.setStyleSheet("color: #8ea0b5; font-size: 11px;")
        layout.addWidget(self.id_label)
        layout.addLayout(lights_row)
        layout.addWidget(self.queue_label)

    def update_data(self, node: dict):
        tag = " (CAMERA)" if node["is_camera_node"] else ""
        self.id_label.setText(f"Node {node['id']}{tag}")
        colors = {"GREEN":"#35d07f", "YELLOW":"#ffd43b", "RED":"#ff5c5c"}
        ns_color, ew_color = colors[node['ns']], colors[node['ew']]
        self.dot_ns.setStyleSheet(f"background:{ns_color}; border-radius:7px;")
        self.dot_ew.setStyleSheet(f"background:{ew_color}; border-radius:7px;")
        self.queue_label.setText(
            f"NS {node['queue_ns']} · EW {node['queue_ew']} · {node['time_in_phase']}s")


class QueueChart(QWidget):
    """Tiny inline line chart of recent total-queue history, drawn directly
    with QPainter — no chart library dependency needed for something this
    small."""
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(90)
        self.history = []

    def set_history(self, history):
        self.history = history
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#17202c"))
        if len(self.history) < 2:
            return
        values = [p["queue"] for p in self.history]
        vmax = max(1, max(values))
        pen = QPen(QColor("#4fd1ff"), 2)
        painter.setPen(pen)
        n = len(values)
        points = []
        for i, v in enumerate(values):
            x = 4 + (i / (n - 1)) * (w - 8)
            y = h - 4 - (v / vmax) * (h - 8)
            points.append((x, y))
        for i in range(1, len(points)):
            painter.drawLine(int(points[i - 1][0]), int(points[i - 1][1]),
                              int(points[i][0]), int(points[i][1]))


def stat_widget(value_text="0", label_text=""):
    box = QWidget()
    box.setStyleSheet("background:#17202c; border-radius:8px;")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(6, 6, 6, 6)
    val = QLabel(value_text); val.setObjectName("statVal"); val.setAlignment(Qt.AlignCenter)
    lbl = QLabel(label_text); lbl.setObjectName("statLbl"); lbl.setAlignment(Qt.AlignCenter); lbl.setWordWrap(True)
    layout.addWidget(val)
    layout.addWidget(lbl)
    return box, val


class FlowMindLiveWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlowMind Live")
        self.resize(1280, 900)

        self.config = load_config()

        self.cameras = CameraManager(self.config)
        self.controller = LiveTrafficController(
            rows=self.config['grid_rows'], cols=self.config['grid_cols'],
            dqn_episodes=self.config['dqn_episodes'], get_live_queues=self.cameras.live_queues,
            get_camera_health=self.cameras.health, control_config=self.config['control'])
        self.cameras.set_signal_callback(self.controller.get_camera_signal)
        self._camera_jobs = queue.Queue()
        self._camera_busy = False
        self._intersection_cards = {}

        self._build_ui()

        self.controller.start()
        self._camera_job(self.cameras.start)

        # Video redraws as fast as new frames arrive — no artificial cap.
        # get_latest_frame() is cheap (a locked copy), so polling at
        # ~33fps ceiling costs little even when the source is idle.
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._refresh_video)
        self.video_timer.start(30)

        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._refresh_stats)
        self.stats_timer.start(500)

        self._intersection_cards = {}

    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setStyleSheet(DARK_STYLE)
        root = QHBoxLayout(self)

        # ---- left column: video + camera settings ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_inner = QWidget()
        left = QVBoxLayout(left_inner)
        left.setSizeConstraint(QLayout.SetMinimumSize)
        left_scroll.setWidget(left_inner)

        video_box = QGroupBox("Four cameras — one intersection (direct video)")
        video_layout = QVBoxLayout(video_box)
        camera_grid = QGridLayout()
        self.camera_inputs, self.camera_approaches, self.camera_enabled = {}, {}, {}
        self.video_labels, self.camera_status_labels = {}, {}
        for i, c in enumerate(self.config['cameras']):
            cid = c['id']
            box = QGroupBox(f"Camera {cid}")
            layout = QVBoxLayout(box)
            video = QLabel("Unused camera")
            video.setAlignment(Qt.AlignCenter)
            video.setMinimumSize(200, 112)
            video.setMaximumHeight(220)
            video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
            video.setStyleSheet("background:#000; color:#8ea0b5;")
            self.video_labels[cid] = video
            layout.addWidget(video)
            status = QLabel("Stopped"); status.setWordWrap(True)
            self.camera_status_labels[cid] = status
            layout.addWidget(status)
            enabled = QCheckBox("Enabled"); enabled.setChecked(c['enabled'])
            self.camera_enabled[cid] = enabled
            layout.addWidget(enabled)
            approach = QComboBox()
            for label, value in [('North','N'),('South','S'),('East','E'),('West','W'),('Whole intersection','ALL')]:
                approach.addItem(label, value)
            approach.setCurrentIndex(approach.findData(c['approach']))
            self.camera_approaches[cid] = approach
            layout.addWidget(approach)
            source = QLineEdit(c['rtsp_url'])
            source.setPlaceholderText("RTSP URL / video file / webcam 0")
            source.setMinimumWidth(0)
            self.camera_inputs[cid] = source
            layout.addWidget(source)
            buttons = QHBoxLayout()
            for action in ('start', 'stop'):
                btn = QPushButton(action.title())
                btn.clicked.connect(lambda checked=False, cid=cid, action=action: self._camera_job(lambda: getattr(self.cameras, action)(cid)))
                buttons.addWidget(btn)
            layout.addLayout(buttons)
            camera_grid.addWidget(box, i // 2, i % 2)
        video_layout.addLayout(camera_grid)
        toolbar_row = QHBoxLayout()
        connect_btn = QPushButton("Save & connect")
        connect_btn.clicked.connect(self._on_connect)
        toolbar_row.addWidget(connect_btn)
        for action in ('start', 'stop'):
            btn = QPushButton(action.title() + ' all')
            btn.clicked.connect(lambda checked=False, action=action: self._camera_job(getattr(self.cameras, action)))
            toolbar_row.addWidget(btn)
        video_layout.addLayout(toolbar_row)
        self.camera_notice = QLabel("One camera per approach. Whole intersection must be used alone.")
        self.camera_notice.setWordWrap(True)
        video_layout.addWidget(self.camera_notice)
        left.addWidget(video_box, stretch=3)

        quality_box = QGroupBox("Camera Quality & Tracking")
        quality_layout = QGridLayout(quality_box)

        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_CHOICES.keys()))
        current_model_label = next((k for k, v in MODEL_CHOICES.items()
                                     if v == self.config["model_name"]), "Fast (nano)")
        self.model_combo.setCurrentText(current_model_label)

        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["480", "640", "960", "1280"])
        self.imgsz_combo.setCurrentText(str(self.config["imgsz"]))

        self.transport_combo = QComboBox()
        self.transport_combo.addItems(["tcp", "udp"])
        self.transport_combo.setCurrentText(self.config["rtsp_transport"])

        self.enhance_check = QCheckBox("Sharpen / boost contrast before detection")
        self.enhance_check.setChecked(self.config["enhance"])

        apply_btn = QPushButton("Apply quality settings")
        apply_btn.clicked.connect(self._on_apply_quality)

        quality_layout.addWidget(QLabel("Detection model"), 0, 0)
        quality_layout.addWidget(self.model_combo, 1, 0)
        quality_layout.addWidget(QLabel("Detection resolution"), 0, 1)
        quality_layout.addWidget(self.imgsz_combo, 1, 1)
        quality_layout.addWidget(QLabel("RTSP transport"), 2, 0)
        quality_layout.addWidget(self.transport_combo, 3, 0)
        quality_layout.addWidget(self.enhance_check, 4, 0, 1, 2)
        quality_layout.addWidget(apply_btn, 5, 0, 1, 2)

        left.addWidget(quality_box, stretch=1)
        root.addWidget(left_scroll, stretch=6)

        # ---- right column: signal, intersections, stats ----
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(510)
        right_inner = QWidget()
        right = QVBoxLayout(right_inner)
        right.setSizeConstraint(QLayout.SetMinimumSize)
        right_scroll.setWidget(right_inner)

        signal_box = QGroupBox("Recommended Signal — right now")
        signal_box.setMinimumHeight(175)
        signal_layout = QVBoxLayout(signal_box)
        dots_row = QHBoxLayout()

        ns_col = QVBoxLayout()
        ns_col.addWidget(QLabel("NORTH / SOUTH"), alignment=Qt.AlignCenter)
        self.light_ns = SignalLight()
        ns_col.addWidget(self.light_ns, alignment=Qt.AlignCenter)
        self.state_ns_label = QLabel("—"); self.state_ns_label.setAlignment(Qt.AlignCenter)
        ns_col.addWidget(self.state_ns_label)

        ew_col = QVBoxLayout()
        ew_col.addWidget(QLabel("EAST / WEST"), alignment=Qt.AlignCenter)
        self.light_ew = SignalLight()
        ew_col.addWidget(self.light_ew, alignment=Qt.AlignCenter)
        self.state_ew_label = QLabel("—"); self.state_ew_label.setAlignment(Qt.AlignCenter)
        ew_col.addWidget(self.state_ew_label)

        dots_row.addLayout(ns_col)
        dots_row.addLayout(ew_col)
        signal_layout.addLayout(dots_row)
        self.signal_meta_label = QLabel("waiting for data…")
        self.signal_meta_label.setAlignment(Qt.AlignCenter)
        self.signal_meta_label.setWordWrap(True)
        self.signal_meta_label.setStyleSheet("color:#8ea0b5; font-size:12px;")
        signal_layout.addWidget(self.signal_meta_label)
        right.addWidget(signal_box)
        control_box = QGroupBox("Control mode & fallback")
        control_box.setMinimumHeight(430)
        controls = QVBoxLayout(control_box)
        mode_row = QHBoxLayout()
        self.mode_buttons = {}
        for mode, title in [('ai', 'AI / retry AI'), ('manual', 'Switch to manual'), ('fixed', 'Fixed timer')]:
            btn = QPushButton(title)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked=False, mode=mode: self._control_command({'mode':mode}))
            self.mode_buttons[mode] = btn
            mode_row.addWidget(btn)
        controls.addLayout(mode_row)
        self.control_status = QLabel(""); self.control_status.setWordWrap(True)
        controls.addWidget(self.control_status)
        manual_row = QHBoxLayout()
        self.manual_buttons = []
        for phase, title in [('NS_GREEN','N/S green'), ('EW_GREEN','E/W green'), ('ALL_RED','Hold all red')]:
            btn = QPushButton(title)
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked=False, phase=phase: self._control_command({'manual_phase':phase}))
            self.manual_buttons.append(btn)
            manual_row.addWidget(btn)
        controls.addLayout(manual_row)
        timer_grid = QGridLayout()
        self.timer_inputs = {}
        for i, (key, label, low, high) in enumerate([
            ('fixed_ns_seconds','N/S green (s)',7,60), ('fixed_ew_seconds','E/W green (s)',7,60),
            ('yellow_seconds','Yellow (s)',1,10), ('all_red_seconds','All red (s)',1,10)]):
            field = QSpinBox(); field.setMinimumHeight(30); field.setRange(low,high); field.setValue(self.config['control'][key])
            self.timer_inputs[key] = field
            timer_grid.addWidget(QLabel(label), i//2*2, i%2)
            timer_grid.addWidget(field, i//2*2+1, i%2)
        controls.addLayout(timer_grid)
        apply_timers = QPushButton("Apply timers")
        apply_timers.clicked.connect(lambda: self._control_command({k:v.value() for k,v in self.timer_inputs.items()}))
        controls.addWidget(apply_timers)
        self.command_message = QLabel(""); self.command_message.setWordWrap(True)
        controls.addWidget(self.command_message)
        note = QLabel("Manual holds until changed. Minimum green: 7s. Changes pass through yellow and all-red. Controls apply to all displayed nodes. AI faults latch fixed-timer fallback. Demo only: no physical signal connection.")
        note.setWordWrap(True)
        controls.addWidget(note)
        right.addWidget(control_box)


        vehicles_box = QGroupBox("Detected Vehicles (all cameras)")
        vgrid = QGridLayout(vehicles_box)
        self.stat_widgets = {}
        for i, key in enumerate(["car", "motorcycle", "bus", "truck"]):
            w, val = stat_widget("0", key)
            vgrid.addWidget(w, 0, i)
            self.stat_widgets[key] = val
        for i, key in enumerate(["in_count", "out_count", "total_count", "capture_fps"]):
            w, val = stat_widget("0", key.replace("_", " "))
            vgrid.addWidget(w, 1, i)
            self.stat_widgets[key] = val
        right.addWidget(vehicles_box)

        self.intersections_box = QGroupBox("Intersections")
        self.intersections_layout = QGridLayout(self.intersections_box)
        right.addWidget(self.intersections_box)

        efficiency_box = QGroupBox("Simulation efficiency (includes synthetic neighbors)")
        elayout = QVBoxLayout(efficiency_box)
        egrid = QGridLayout()
        for i, key in enumerate(["avg_wait_per_vehicle_s", "avg_queue_recent",
                                  "vehicles_served", "phase_switches"]):
            w, val = stat_widget("0", key.replace("_", " "))
            egrid.addWidget(w, 0, i)
            self.stat_widgets[key] = val
        elayout.addLayout(egrid)
        self.chart = QueueChart()
        elayout.addWidget(self.chart)
        self.training_label = QLabel("")
        self.training_label.setStyleSheet("color:#8ea0b5; font-size:12px;")
        self.training_label.setWordWrap(True)
        elayout.addWidget(self.training_label)
        right.addWidget(efficiency_box)

        right.addStretch()
        root.addWidget(right_scroll, stretch=4)

    # ------------------------------------------------------------------
    def _camera_job(self, action):
        if self._camera_busy:
            self.camera_notice.setText("A camera operation is still running. Light controls remain available.")
            return
        self._camera_busy = True
        self.camera_notice.setText("Updating cameras… light controls remain available.")
        def work():
            try:
                action()
                self._camera_jobs.put("Camera operation completed.")
            except Exception as error:
                self._camera_jobs.put(str(error))
        threading.Thread(target=work, daemon=True).start()

    def _apply_camera_settings(self, data, start=False):
        try:
            updated = update_settings(self.config, data)
        except ValueError as error:
            self.camera_notice.setText(str(error))
            return
        def work():
            self.cameras.configure(updated)
            if start:
                self.cameras.start()
        # Save on the UI thread; slow camera work runs separately. Control
        # commands can persist newer control settings without stale overwrite.
        if not self._camera_busy:
            self.config = updated
            save_config(self.config)
        else:
            self.camera_notice.setText("Wait for the current camera operation to finish.")
            return
        self._camera_job(work)

    def _on_connect(self):
        self._apply_camera_settings({'cameras':[
            dict(id=cid, rtsp_url=field.text().strip(),
                 approach=self.camera_approaches[cid].currentData(), enabled=self.camera_enabled[cid].isChecked())
            for cid, field in self.camera_inputs.items()]}, start=True)

    def _on_apply_quality(self):
        quality = {'yolov8n.pt':'fast','yolov8s.pt':'balanced','yolov8m.pt':'accurate'}[MODEL_CHOICES[self.model_combo.currentText()]]
        self._apply_camera_settings({'model_quality':quality, 'imgsz':int(self.imgsz_combo.currentText()),
            'rtsp_transport':self.transport_combo.currentText(), 'enhance':self.enhance_check.isChecked()})

    def _control_command(self, data):
        try:
            self.controller.configure(data)
            self.config['control'] = dict(self.controller.control)
            save_config(self.config)
            self.command_message.setText("Command accepted; yellow/all-red clearance is preserved.")
            self._refresh_stats()
        except (ValueError, OSError) as error:
            self.command_message.setText(str(error))

    def _refresh_video(self):
        for cid, label in self.video_labels.items():
            vision = self.cameras.get(cid)
            frame = vision.get_latest_frame() if vision._running else None
            if frame is None:
                label.setText("Waiting / stopped")
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refresh_stats(self):
        while not self._camera_jobs.empty():
            self.camera_notice.setText(self._camera_jobs.get_nowait())
            self._camera_busy = False
        for c in self.cameras.snapshot():
            status = f"{c['approach']} · {c['camera_status']} · {c['fps']} FPS"
            if c['configured'] and not c['healthy']:
                status += ' · ' + (c['last_error'] or 'waiting / stale')
            self.camera_status_labels[c['id']].setText(status if c['configured'] else 'Unused camera')
        v = self.cameras.aggregate()
        t = self.controller.snapshot()
        control = t['control']
        for mode, button in self.mode_buttons.items():
            button.setChecked(control['effective_mode'] == mode)
        for button in self.manual_buttons:
            button.setEnabled(control['effective_mode'] == 'manual')
        self.control_status.setText(('FIXED TIMER FALLBACK: ' + control['fallback_reason'] + '. Select AI / retry AI to resume.')
            if control['fallback_active'] else ('Active mode: ' + control['effective_mode'].upper() +
                (' · ' + control['manual_phase'] if control['effective_mode'] == 'manual' else '')))
        self.control_status.setStyleSheet('color:#ffb545;' if control['fallback_active'] else 'color:#8ea0b5;')

        for key in ("car", "motorcycle", "bus", "truck"):
            self.stat_widgets[key].setText(str(v["vehicle_counts"].get(key, 0)))
        self.stat_widgets["in_count"].setText(str(v["in_count"]))
        self.stat_widgets["out_count"].setText(str(v["out_count"]))
        self.stat_widgets["total_count"].setText(str(v["total_count"]))
        self.stat_widgets["capture_fps"].setText(str(v["capture_fps"]))

        sig = t["camera_signal"]
        self.light_ns.set_state(sig["ns"])
        self.light_ew.set_state(sig["ew"])
        self.state_ns_label.setText(sig["ns"])
        self.state_ew_label.setText(sig["ew"])
        self.signal_meta_label.setText(
            f"NS queue {sig['queue_ns']} · EW queue {sig['queue_ew']} · "
            f"{sig['time_in_phase']}s · clearance {sig['transition_remaining_s']}s · next {sig['target']}")

        for node in t["intersections"]:
            nid = node["id"]
            if nid not in self._intersection_cards:
                card = IntersectionCard()
                self._intersection_cards[nid] = card
                idx = len(self._intersection_cards) - 1
                self.intersections_layout.addWidget(card, idx // 2, idx % 2)
            self._intersection_cards[nid].update_data(node)

        eff = t["efficiency"]
        self.stat_widgets["avg_wait_per_vehicle_s"].setText(f"{eff['avg_wait_per_vehicle_s']}s")
        self.stat_widgets["avg_queue_recent"].setText(str(eff["avg_queue_recent"]))
        self.stat_widgets["vehicles_served"].setText(str(eff["vehicles_served"]))
        self.stat_widgets["phase_switches"].setText(str(eff["phase_switches"]))
        self.chart.set_history(t["history"][-120:])
        self.training_label.setText(f"Controller: {t['controller']} — {t['training_status']}")

    def closeEvent(self, event):
        self.controller.stop()
        threading.Thread(target=self.cameras.stop, daemon=True).start()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = FlowMindLiveWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
