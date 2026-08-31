"""Four independent trackers feeding ONE intersection, one approach per view."""
import threading
import time
from live_settings import normalize_cameras

QUALITY_KEYS = ('confidence', 'line_position', 'model_name', 'imgsz', 'rtsp_transport', 'enhance')


class CameraManager:
    def __init__(self, config, signal_callback=None, factory=None, clock=time.monotonic):
        if factory is None:
            from vision import VehicleVision
            factory = VehicleVision
        self.factory, self.clock = factory, clock
        self.signal_callback = signal_callback
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self.config = config
        self.items = normalize_cameras(config['cameras'])
        self.visions = {c['id']: self._make(c, config) for c in self.items}
        self.started = {}

    def _make(self, item, config):
        zones = None if item['approach'] == 'ALL' else {
            d: ((0., 0., 1., 1.) if d == item['approach'] else (0., 0., 0., 0.))
            for d in ('N', 'S', 'E', 'W')}
        return self.factory(rtsp_url=item['rtsp_url'], zones=zones,
                            get_signal_state=self.signal_callback,
                            **{k: config[k] for k in QUALITY_KEYS})

    def set_signal_callback(self, callback):
        self.signal_callback = callback
        for v in self.visions.values():
            v.get_signal_state = callback

    def get(self, camera_id='N'):
        with self._lock:
            if camera_id not in self.visions:
                raise ValueError('Unknown camera; use N, S, E or W')
            return self.visions[camera_id]

    def configure(self, config):
        items = normalize_cameras(config['cameras'])
        with self._operation_lock:
            for item in items:
                cid = item['id']
                old = next(c for c in self.items if c['id'] == cid)
                changed = item != old or any(config[k] != self.config[k] for k in QUALITY_KEYS)
                if changed:
                    previous = self.get(cid)
                    was_running = previous._running
                    replacement = self._make(item, config)
                    with self._lock:
                        self.visions[cid] = replacement
                        self.started.pop(cid, None)
                    # The old object's threads can never publish into the new
                    # camera's state, even if its RTSP read takes time to stop.
                    previous.stop()
                    if was_running and item['enabled'] and item['rtsp_url']:
                        replacement.start()
                        self.started[cid] = self.clock()
            with self._lock:
                self.items, self.config = items, config

    def start(self, camera_id=None):
        with self._operation_lock:
            if camera_id is not None:
                self.get(camera_id)
            for c in self.items:
                if camera_id is not None and c['id'] != camera_id:
                    continue
                if c['enabled'] and c['rtsp_url']:
                    v = self.get(c['id'])
                    if not v._running:
                        v.start()
                        self.started[c['id']] = self.clock()

    def stop(self, camera_id=None):
        with self._operation_lock:
            if camera_id is not None:
                self.get(camera_id)
            for c in self.items:
                if camera_id is None or c['id'] == camera_id:
                    self.started.pop(c['id'], None)
                    self.get(c['id']).stop()

    def snapshot(self):
        with self._lock:
            now = self.clock()
            result = []
            for c in self.items:
                v = self.visions[c['id']]
                state = v.state.snapshot()
                age = state.get('last_detection_age_s')
                healthy = (v._running and state['camera_status'].startswith('online') and
                           state.get('detection_ok', False) and age is not None and age <= 5)
                configured = c['enabled'] and bool(c['rtsp_url'])
                result.append({**c, **state, 'configured': configured, 'healthy': bool(healthy),
                    'starting': c['id'] in self.started and now - self.started[c['id']] < 8,
                    'running': v._running})
            return result

    def health(self):
        configured = [s for s in self.snapshot() if s['configured']]
        bad = [s['id'] for s in configured if not s['healthy'] and not s['starting']]
        return {'configured': len(configured), 'healthy': not bad,
                'reason': 'Camera/detection unavailable or stale: ' + ', '.join(bad) if bad else ''}

    def live_queues(self):
        queues = dict.fromkeys(('N', 'S', 'E', 'W'), 0)
        for c in self.snapshot():
            if not c['configured'] or not c['healthy']:
                continue
            if c['approach'] == 'ALL':
                queues.update(c['queues'])
            else:
                queues[c['approach']] = c['queues'][c['approach']]
        return queues

    def aggregate(self):
        states = [c for c in self.snapshot() if c['configured']]
        return {'camera_status': f"{sum(c['healthy'] for c in states)}/{len(states)} cameras healthy",
            'vehicle_counts': {k: sum(c['vehicle_counts'][k] for c in states) for k in ('car','motorcycle','bus','truck')},
            **{k: sum(c[k] for c in states) for k in ('in_count','out_count','total_count','fps','capture_fps')},
            'queues': self.live_queues()}
